#!/usr/bin/env python3
"""Auto-export the robot URDF for Pinocchio.

Pinocchio can't read xacro, so the model must be flattened to a plain
URDF first. This used to be a manual step every session:

    xacro .../ur.urdf.xacro ur_type:=ur3e name:=ur > /tmp/ur3e.urdf

Now ``load_pinocchio()`` calls :func:`ensure_urdf` automatically when
the file is missing, so nodes just work in a fresh shell. Delete the
file to force a re-export after changing the robot description:

    rm /tmp/ur3e.urdf

This module needs the workspace to be sourced (it locates
``ur_description`` via ament) and the ``xacro`` package — both are
already requirements of running any node here. It deliberately does
NOT import rclpy, so the kinematics helpers stay usable outside ROS.

Author: Tejas
"""

import os

#: xacro arguments used for this project's robot. The vendored
#: ur_description's ur_macro.xacro also pulls in the RH-P12-RN gripper,
#: which provides the rh_p12_rn_ee TCP frame the IK plans to.
DEFAULT_XACRO_ARGS = {
    'ur_type': 'ur3e',
    'name': 'ur',
}


def ensure_urdf(urdf_path: str) -> str:
    """Make sure a flattened URDF exists at ``urdf_path``.

    If the file already exists it is left untouched (delete it to force
    a re-export). Otherwise the project's default robot description
    (ur3e + gripper) is processed with xacro and written there.

    Args:
        urdf_path: Destination path, e.g. ``/tmp/ur3e.urdf``.

    Returns:
        ``urdf_path``, for convenient chaining.

    Raises:
        RuntimeError: If xacro/ament are unavailable (workspace not
            sourced?) or processing fails.
    """
    if os.path.exists(urdf_path):
        return urdf_path

    try:
        import xacro
        from ament_index_python.packages import get_package_share_directory
    except ImportError as e:
        raise RuntimeError(
            f'URDF missing at {urdf_path} and auto-export unavailable '
            f'({e}). Source the workspace, or export manually: '
            f'xacro .../ur.urdf.xacro ur_type:=ur3e name:=ur > {urdf_path}'
        ) from e

    xacro_file = os.path.join(
        get_package_share_directory('ur_description'), 'urdf', 'ur.urdf.xacro')

    doc = xacro.process_file(xacro_file, mappings=dict(DEFAULT_XACRO_ARGS))
    urdf_xml = doc.toprettyxml(indent='  ')

    # Write atomically-ish: a half-written file would poison every later
    # session, since ensure_urdf only regenerates when the file is absent.
    tmp_path = urdf_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(urdf_xml)
    os.replace(tmp_path, urdf_path)
    return urdf_path
