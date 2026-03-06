#!/usr/bin/env python3
"""
Create GitHub issues for all stories in the decomposition.
This script generates issues programmatically.
"""
import subprocess
import os
import re

# GitHub repo information
REPO = "jarvanclaw-dev/perf-pressure-traverse"
GH_TOKEN = os.getenv("GH_TOKEN", "")

stories = {
    "1": {
        "title": "Establish project structure and package skeleton",
        "description": "Set up the project repository with complete directory structure, configuration files, and package initialization. Establish a clean, production-ready Python package structure following best practices.\n\n**Technical Context:**\n- Module: core, main package\n- Dependencies: None (foundation)  \n- Related Tasks: All subsequent implementation stories",
        "labels": ["epic", "story", "p0-critical", "forge"],
        "assignee": "forge"
    },
    "2": {
        "title": "Implement domain model classes",
        "description": "Create dataclasses for all domain models: `FluidProperties`, `WellGeometry`, `PVTProperties`, and `PressureTraverseResult`. Implement property methods for computed properties and basic validation in the data classes.\n\n**Technical Context:**\n- Module: models/\n- Dependencies: Story 1\n- Related Tasks: Validation stories",
        "labels": ["epic", "story", "p0-critical", "forge"],
        "assignee": "forge"
    },
    "3": {
        "title": "Implement unit conversion utilities",
        "description": "Create comprehensive unit conversion functions for API units (ft, psi, °F, °R) and SI units (m, Pa, K). Ensure all conversions are consistent and well-tested.\n\n**Technical Context:**\n- Module: core/units.py\n- Dependencies: Story 1  \n- Related Tasks: Validation modules",
        "labels": ["epic", "story", "p2-medium", "forge"],
        "assignee": "forge"
    },
    "4": {
        "title": "Create parameter validation and error handling framework",
        "description": "Implement comprehensive validation utilities for fluid properties, well geometry, and calculation parameters. Define custom exceptions and validation rules to catch physical inconsistencies early.\n\n**Technical Context:**\n- Module: utils/validation.py\n- Dependencies: Stories 1, 2\n- Related Tasks: Validation testing",
        "labels": ["epic", "story", "p1-high", "forge"],
        "assignee": "forge"
    },
    "5": {
        "title": "Create validation testing framework",
        "description": "Build framework for validating all parameter inputs with test-driven approach. Use pytest fixtures for common test parameters and parametrize validation tests.\n\n**Technical Context:**\n- Module: tests/validation/\n- Dependencies: Stories 2, 4\n- Related Tasks: Foundation testing",
        "labels": ["epic", "story", "p1-high", "forge"],
        "assignee": "forge"
    },
    "6": {
        "title": "Implement black oil PVT property correlations (Vapor pressure, BPF, density)",
        "description": "Implement key black oil correlations: Standing vapor pressure, formation volume factors, solution gas ratio, and density calculations. Use standard correlations (Standing, Vasquez-Beggs, Glaso) per API RPI.\n\n**Technical Context:**\n- Module: models/fluid.py\n- Dependencies: Story 2\n- Related Tasks: Z-factor stories",
        "labels": ["epic", "story", "PVT", "p1-high", "forge"],
        "assignee": "forge"
    },
    "7": {
        "title": "Implement natural gas Z-Factor models",
        "description": "Implement multiple Z-factor correlations (Standing-Katz, Lee-Gonzales, AGA-DC). Create reusable implementation for Z-factor tables and interpolation, supporting both natural gas and gas-condensate systems.\n\n**Technical Context:**\n- Module: math/z_factor.py\n- Dependencies: Story 6\n- Related Tasks: EOS stories",
        "labels": ["epic", "story", "PVT", "p1-high", "forge"],
        "assignee": "forge"
    },
    "8": {
        "title": "Implement Equation of State (EOS) solvers for compositional systems",
        "description": "Create abstract EOS base class and implement SRK and Peng-Robinson equations with cubic solving. This will support advanced gas-condensate systems requiring compositional calculations.\n\n**Technical Context:**\n- Module: math/eos.py\n- Dependencies: Story 7\n- Related Tasks: Advanced PVT",
        "labels": ["epic", "story", "PVT", "p2-medium", "Advanced", "forge"],
        "assignee": "forge"
    },
    "9": {
        "title": "Implement flow regime identification logic",
        "description": "Create flow regime detection module that identifies flow regime types (Segregated, Intermittent, Distributed, Mist) based on fluid properties and well geometry. Implement Beggs-Brill flow regime transition map.\n\n**Technical Context:**\n- Module: flow/regime.py\n- Dependencies: Story 2\n- Related Tasks: Correlation implementations",
        "labels": ["epic", "story", "flow", "p1-high", "forge"],
        "assignee": "forge"
    },
    "10": {
        "title": "Implement friction factor models (Darcy-Weisbach, Moody, API)",
        "description": "Implement friction factor calculations for multiphase flow: Moody diagram lookup, API correlation, and Darcy-Weisbach equation integration. Support laminar, transitional, and turbulent regimes.\n\n**Technical Context:**\n- Module: flow/friction.py\n- Dependencies: Story 3\n- Related Tasks: Beggs-Brill correlation",
        "labels": ["epic", "story", "flow", "p1-high", "forge"],
        "assignee": "forge"
    },
    "11": {
        "title": "Implement Beggs & Brill (1973) multiphase flow correlation",
        "description": "Fully implement Beggs-Brill (1973) correlation for pressure drop, liquid holdup, and friction factor in pipes. This is the industry standard for multiphase flow calculations.\n\n**Technical Context:**\n- Module: flow/correlations.py\n- Dependencies: Stories 3, 4, 6, 7, 8, 9, 10\n- Related Tasks: Additional correlations",
        "labels": ["epic", "story", "flow", "p1-high", "forge", "critical"],
        "assignee": "forge"
    },
    "12": {
        "title": "Implement Hagedorn-Brown and Gray correlations",
        "description": "Implement Hagedorn-Brown (optimized for vertical flow) and Gray (optimized for gas wells) correlations for pressure traverse calculations.\n\n**Technical Context:**\n- Module: flow/correlations.py\n- Dependencies: Story 11\n- Related Tasks: Additional flow models",
        "labels": ["epic", "story", "flow", "p2-medium", "forge"],
        "assignee": "forge"
    },
    "13": {
        "title": "Integrate Z-factor with flow correlation calculations",
        "description": "Connect Z-factor calculations with flow correlations and pressure solver. Ensure gas compressibility is correctly accounted for in pressure gradient calculations.\n\n**Technical Context:**\n- Integration module: flow/\n- Dependencies: Stories 3, 7, 11\n- Related Tasks: Core solver",
        "labels": ["epic", "story", "flow", "p1-high", "forge"],
        "assignee": "forge"
    },
    "14": {
        "title": "Implement Newton-Raphson pressure solver",
        "description": "Develop Newton-Raphson iterative solver for nonlinear pressure traverse equations. Implement convergence criteria, error handling, and diagnostics for iterative solver stability.\n\n**Technical Context:**\n- Module: core/pressure_traverse.py (solver) \n- Module: math/iterative.py\n- Dependencies: Story 4, Story 14 (self-dependency)\n- Related Tasks: Main solver",
        "labels": ["epic", "story", "core", "p0-critical", "forge"],
        "assignee": "forge"
    },
    "15": {
        "title": "Implement main pressure traverse solver (surface-to-bottom and bottom-to-surface)",
        "description": "Build the complete `PressureTraverseSolver` class with sweep algorithms and stepwise pressure traversals. Integrate all components (PVT, correlations, friction, solver). Implement surface-to-bottom and bottom-to-surface traverses.\n\n**Technical Context:**\n- Module: core/pressure_traverse.py\n- Dependencies: Stories 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14\n- Related Tasks: All previously implemented components",
        "labels": ["epic", "story", "core", "p0-critical", "forge", "critical"],
        "assignee": "forge"
    },
    "16": {
        "title": "Implement API Recommended Practice 14A (1976) test cases",
        "description": "Create comprehensive test suite with all API RPI standard test cases. Implement test data structures and automated validation against known correct answers.\n\n**Technical Context:**\n- Module: tests/api/\n- Dependencies: Story 15\n- Related Tasks: Validation testing",
        "labels": ["epic", "story", "testing", "p0-critical", "forge"],
        "assignee": "forge"
    },
    "17": {
        "title": "Verify numerical accuracy and handle edge cases",
        "description": "Comprehensive testing for numerical stability, edge cases, and accuracy. Test extreme conditions, transition zones, and convergence scenarios.\n\n**Technical Context:**\n- Module: tests/\n- Dependencies: Story 16\n- Related Tasks: Test quality",
        "labels": ["epic", "story", "testing", "p1-high", "forge"],
        "assignee": "forge"
    },
    "18": {
        "title": "Generate comprehensive API documentation and usage examples",
        "description": "Create Sphinx documentation with API reference, installation guide, and usage examples. Document all public APIs, parameters, return values, and error conditions.\n\n**Technical Context:**\n- Module: docs/\n- Dependencies: Story 15\n- Related Tasks: User adoption",
        "labels": ["epic", "story", "documentation", "p2-medium", "forge"],
        "assignee": "forge"
    },
    "19": {
        "title": "Create Docker container and deployment documentation",
        "description": "Create Dockerfile and docker-compose.yml for reproducible environment. Provide installation instructions for various platforms and deployment guidance.\n\n**Technical Context:**\n- Module: Dockerfile, docker-compose.yml\n- Dependencies: Story 15\n- Related Tasks: Deployment",
        "labels": ["epic", "story", "deployment", "p1-high", "forge"],
        "assignee": "forge"
    },
    "20": {
        "title": "Set up CI/CD pipeline with automated testing",
        "description": "Configure GitHub Actions for continuous integration with automated builds, testing, and documentation preview.\n\n**Technical Context:**\n- Module: .github/workflows/\n- Dependencies: All testing and documentation stories\n- Related Tasks: Development workflow",
        "labels": ["epic", "story", "deployment", "p1-high", "forge"],
        "assignee": "forge"
    }
}

def format_description(desc):
    """Format description with Markdown."""
    return desc.strip().replace('\n\n', '\n\n')

def create_github_issue(issue_num, story_data):
    """Create a GitHub issue using the gh CLI."""
    title = story_data["title"]
    description = format_description(story_data["description"])
    labels = story_data["labels"]
    assignee = story_data.get("assignee", "")

    import json
    issue_data = {
        "title": title,
        "body": description,
        "labels": labels
    }

    if assignee:
        issue_data["assignees"] = [assignee]

    result = subprocess.run(
        ["gh", "issue", "create", "--repo", REPO],
        input=json.dumps(issue_data),
        capture_output=True,
        text=True
    )

    return result.returncode, result.stdout, result.stderr

def main():
    """Create issues for all stories."""
    print(f"Creating GitHub issues for {len(stories)} stories...")
    print(f"GitHub repo: {REPO}")
    print()

    success_count = 0
    fail_count = 0

    for issue_num, story_data in stories.items():
        returncode, stdout, stderr = create_github_issue(issue_num, story_data)

        if returncode == 0:
            print(f"✓ Issue #{issue_num} created")
            print(f"  {story_data['title']}")
            print()
            success_count += 1
        else:
            print(f"✗ Issue #{issue_num} failed")
            print(f"  {story_data['title']}")
            print(f"  Error: {stderr}")
            print()
            fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Issues created: {success_count}")
    print(f"  Issues failed: {fail_count}")
    print(f"{'=' * 60}")

    if success_count > 0:
        print(f"\nView all issues at: https://github.com/{REPO}/issues")

if __name__ == "__main__":
    main()