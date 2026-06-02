import argparse
import subprocess

def main() -> None:
    parser = argparse.ArgumentParser(description='Serve RL dashboard as a standalone process')
    parser.add_argument('--port', type=int, default=8766, help='HTTP port for dashboard server')
    parser.add_argument('--directory', type=str, default='.', help='Directory to serve')
    args = parser.parse_args()

    print(f"[DASHBOARD] Starting streamlit dashboard on port {args.port}...", flush=True)
    subprocess.run(["streamlit", "run", "streamlit_dashboard.py", "--server.port", str(args.port)])

if __name__ == '__main__':
    main()

