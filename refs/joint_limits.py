#!/usr/bin/env python3
"""Script to extract joint limits from URDF files."""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
from xml.etree import ElementTree as ET


@dataclass
class JointLimit:
    """Represents the limits of a joint in radians."""

    name: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None

    def __str__(self) -> str:
        """Format joint limit information as a string."""
        limit_str = f"Joint: {self.name}\n"
        if self.lower is not None:
            limit_str += (
                f"  Lower: {self.lower:.3f} rad ({math.degrees(self.lower):.1f}°)\n"
            )
        if self.upper is not None:
            limit_str += (
                f"  Upper: {self.upper:.3f} rad ({math.degrees(self.upper):.1f}°)\n"
            )
        if self.effort is not None:
            limit_str += f"  Effort: {self.effort:.1f}\n"
        if self.velocity is not None:
            limit_str += f"  Velocity: {self.velocity:.1f}\n"
        return limit_str


def parse_urdf(urdf_path: Path) -> Dict[str, JointLimit]:
    """Parse a URDF file and extract joint limits.

    Args:
        urdf_path: Path to the URDF file.

    Returns:
        Dictionary mapping joint names to their limits.

    Raises:
        FileNotFoundError: If the URDF file doesn't exist.
        ET.ParseError: If the URDF file is invalid XML.
    """
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_limits: Dict[str, JointLimit] = {}

    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name is None:
            continue

        # Skip fixed joints as they don't have limits
        if joint.get("type") == "fixed":
            continue

        limit_elem = joint.find("limit")
        if limit_elem is None:
            joint_limits[name] = JointLimit(name=name)
            continue

        # Convert degree values to radians if present
        lower = limit_elem.get("lower")
        upper = limit_elem.get("upper")
        effort = limit_elem.get("effort")
        velocity = limit_elem.get("velocity")

        joint_limits[name] = JointLimit(
            name=name,
            lower=float(lower) if lower is not None else None,
            upper=float(upper) if upper is not None else None,
            effort=float(effort) if effort is not None else None,
            velocity=float(velocity) if velocity is not None else None,
        )

    return joint_limits


def print_joint_limits(joint_limits: Sequence[JointLimit]) -> None:
    """Print joint limits in a formatted way.

    Args:
        joint_limits: Sequence of JointLimit objects to print.
    """
    print("\nJoint Limits:")
    print("-" * 50)
    for limit in sorted(joint_limits, key=lambda x: x.name):
        print(f"{limit}\n")


def main() -> None:
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Extract joint limits from a URDF file"
    )
    parser.add_argument(
        "urdf_path",
        type=Path,
        help="Path to the URDF file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file path for the results",
    )

    args = parser.parse_args()

    try:
        joint_limits = parse_urdf(args.urdf_path)

        if args.output:
            # Redirect output to file if specified
            with args.output.open("w", encoding="utf-8") as f:
                sys.stdout = f  # type: ignore
                print_joint_limits(list(joint_limits.values()))
                sys.stdout = sys.__stdout__  # type: ignore
        else:
            print_joint_limits(list(joint_limits.values()))

    except (FileNotFoundError, ET.ParseError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
