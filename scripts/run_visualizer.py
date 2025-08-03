#!/usr/bin/env python3
"""
Simple CLI launcher for OpenSearch Vector Visualizer.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='OpenSearch Vector Visualizer Launcher')
    parser.add_argument('--port', type=int, default=8501, help='Port to run Streamlit on')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    main_script = project_root / "opensearch_visualizer" / "main.py"
    
    if not main_script.exists():
        print(f"❌ Main script not found: {main_script}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        "streamlit", "run",
        str(main_script),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    if args.dev:
        cmd.extend([
            "--server.runOnSave", "true",
            "--server.allowRunOnSave", "true"
        ])
    
    print(f"🚀 Starting OpenSearch Vector Visualizer...")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n✅ Visualizer stopped")
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()