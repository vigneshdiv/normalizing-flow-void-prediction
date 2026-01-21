"""
Script to download void data files from Globus for multiple simulations.
Downloads shapes_all and centers_all files for simulations 2-100.
"""
import subprocess
import sys
from pathlib import Path
import time

# Configuration
SOURCE_ENDPOINT_ID = "e0eae0aa-5bca-11ea-9683-0e56c063f437"
START_SIM = 2
END_SIM = 100
DEST_DIR = Path("/Users/vignesh/Documents/VoidData")


def run_command(cmd, check=True, verbose=True):
    """Run a shell command and return the result."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def check_globus_cli():
    """Check if Globus CLI is installed."""
    try:
        # Try a simple command that works with globus CLI
        result = run_command(["globus", "--help"], check=False)
        if result.returncode == 0 or result.returncode == 2:  # --help returns 2, but means CLI exists
            print("Globus CLI found")
            return True
        else:
            print("Globus CLI not found. Please install it with: pip3 install globus-cli")
            return False
    except FileNotFoundError:
        print("Globus CLI not found. Please install it with: pip3 install globus-cli")
        return False


def check_authentication():
    """Check if user is authenticated with Globus."""
    result = run_command(["globus", "whoami"], check=False)
    if result.returncode == 0:
        print(f"Authenticated as: {result.stdout.strip()}")
        return True
    else:
        print("Not authenticated. Please run: globus login")
        return False


def find_destination_endpoint():
    """Find the local destination endpoint ID."""
    print("\n" + "=" * 60)
    print("Finding your local endpoint (where files will be downloaded)")
    print("=" * 60)
    
    # Get current user email to search for their endpoints
    whoami_result = run_command(["globus", "whoami"], check=False)
    user_email = whoami_result.stdout.strip() if whoami_result.returncode == 0 else None
    
    # Search for endpoints owned by the user
    print("\nSearching for your personal endpoints...")
    result = run_command(["globus", "endpoint", "search", "--filter-scope", "my-endpoints"], check=False)
    
    endpoints_found = []
    if result.returncode == 0 and result.stdout.strip():
        lines = result.stdout.strip().split('\n')
        print("\nYour endpoints:")
        for i, line in enumerate(lines[2:], 1):  # Skip header lines
            if line.strip():
                parts = line.split('|')
                if len(parts) >= 3:
                    endpoint_id = parts[0].strip()
                    owner = parts[1].strip()
                    name = parts[2].strip()
                    endpoints_found.append((endpoint_id, name))
                    print(f"  {i}. {name}")
                    print(f"     ID: {endpoint_id}")
    
    if endpoints_found:
        # If only one endpoint found, use it automatically
        if len(endpoints_found) == 1:
            selected = endpoints_found[0]
            print(f"\n✓ Auto-selected endpoint: {selected[1]} ({selected[0]})")
            return selected[0]
        
        # Multiple endpoints - ask user to choose
        print("\n" + "-" * 60)
        choice = input(f"\nSelect an endpoint (1-{len(endpoints_found)}) or press Enter to enter ID manually: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(endpoints_found):
            selected = endpoints_found[int(choice) - 1]
            print(f"\nUsing endpoint: {selected[1]} ({selected[0]})")
            return selected[0]
    
    # If no endpoints found or user wants to enter manually
    print("\n" + "=" * 60)
    print("No personal endpoint found or manual entry requested")
    print("=" * 60)
    print("\nTo set up a local endpoint for your Mac:")
    print("  1. Install Globus Connect Personal: https://www.globus.org/globus-connect-personal")
    print("  2. Run it and it will create a personal endpoint")
    print("  3. Find your endpoint ID:")
    print("     - In Globus Connect Personal app, or")
    print("     - Go to https://app.globus.org/file-manager")
    print("     - Click 'Endpoints' → Look for your personal endpoint")
    print("     - Copy the endpoint ID (looks like: xxxx-xxxx-xxxx-xxxx)")
    print("\nAlternatively, you can use any endpoint you have access to.")
    print("\n" + "-" * 60)
    endpoint_id = input("\nEnter destination endpoint ID: ").strip()
    if not endpoint_id:
        print("\nError: No endpoint ID provided")
        print("Please set up Globus Connect Personal or provide a valid endpoint ID.")
        sys.exit(1)
    
    # Verify the endpoint exists and is accessible
    print(f"\nVerifying endpoint {endpoint_id}...")
    verify_result = run_command(["globus", "endpoint", "show", endpoint_id], check=False)
    if verify_result.returncode != 0:
        print(f"\n❌ Error: Endpoint {endpoint_id} not found or not accessible")
        print(f"Error details: {verify_result.stderr}")
        print("\nPlease:")
        print("  1. Make sure Globus Connect Personal is running (if using a personal endpoint)")
        print("  2. Verify the endpoint ID is correct")
        print("  3. Check https://app.globus.org/file-manager → Endpoints to find your endpoint")
        sys.exit(1)
    
    print(f"✓ Endpoint verified: {endpoint_id}")
    return endpoint_id


def verify_source_paths(source_ep, simulation_num, verbose=True):
    """Verify that source files exist for a given simulation."""
    source_base_path = f"/Gigantes/latin_hypercube/z0.0/sample_Quijote_{simulation_num}_ss1.0_z0.00_d00"
    shapes_file = f"shapes_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    centers_file = f"centers_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    
    if verbose:
        print(f"\nVerifying source directory: {source_base_path}")
    result = run_command([
        "globus", "ls", f"{source_ep}:{source_base_path}"
    ], check=False, verbose=False)
    
    if result.returncode != 0:
        if verbose:
            print(f"Error listing directory: {result.stderr}")
        return False
    
    files = result.stdout.strip().split('\n')
    
    if shapes_file not in files:
        if verbose:
            print(f"Warning: {shapes_file} not found in directory")
        return False
    if centers_file not in files:
        if verbose:
            print(f"Warning: {centers_file} not found in directory")
        return False
    
    if verbose:
        print(f"✓ Both files found: {shapes_file} and {centers_file}")
    return True


def transfer_file(source_ep, dest_ep, simulation_num, file_type):
    """Transfer a single file using Globus.
    
    Args:
        source_ep: Source endpoint ID
        dest_ep: Destination endpoint ID
        simulation_num: Simulation number
        file_type: 'shapes' or 'centers'
    
    Returns:
        Task ID if successful, None otherwise
    """
    source_base_path = f"/Gigantes/latin_hypercube/z0.0/sample_Quijote_{simulation_num}_ss1.0_z0.00_d00"
    
    if file_type == 'shapes':
        source_file = f"shapes_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    elif file_type == 'centers':
        source_file = f"centers_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    else:
        raise ValueError("file_type must be 'shapes' or 'centers'")
    
    dest_file = str(DEST_DIR / source_file)
    source_path = f"{source_ep}:{source_base_path}/{source_file}"
    dest_path = f"{dest_ep}:{dest_file}"
    
    print(f"  Transferring: {source_file}")
    
    result = run_command([
        "globus", "transfer",
        source_path,
        dest_path,
        "--label", f"Download sim{simulation_num} {file_type}"
    ], check=False, verbose=False)
    
    if result.returncode != 0:
        print(f"    ❌ Error: {result.stderr.strip()}")
        return None
    
    # Extract task ID from output
    output = result.stdout
    task_id = None
    for line in output.split('\n'):
        if 'Task ID:' in line:
            task_id = line.split('Task ID:')[1].strip()
            break
    
    if task_id:
        print(f"    ✓ Task ID: {task_id}")
        return task_id
    else:
        print(f"    ⚠ Warning: Could not extract task ID")
        return None


def wait_for_transfer(task_id, verbose=True):
    """Wait for a transfer task to complete."""
    if not task_id:
        return False
    
    if verbose:
        print(f"    Waiting for transfer {task_id} to complete...")
    result = run_command([
        "globus", "task", "wait", task_id
    ], check=False, verbose=False)
    
    if result.returncode == 0:
        if verbose:
            print(f"    ✓ Transfer completed successfully!")
        return True
    else:
        if verbose:
            print(f"    ⚠ Transfer may still be in progress. Check with: globus task show {task_id}")
        return False


def download_simulation(source_ep, dest_ep, simulation_num):
    """Download files for a single simulation."""
    print(f"\n[{simulation_num}/{END_SIM}] Processing simulation {simulation_num}...")
    
    # Verify source paths exist
    if not verify_source_paths(source_ep, simulation_num, verbose=False):
        print(f"  ❌ Simulation {simulation_num}: Files not found, skipping...")
        return False
    
    # Check if files already exist locally
    shapes_file = f"shapes_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    centers_file = f"centers_all_Quijote_{simulation_num}_ss1.0_z0.00_d00.out"
    
    shapes_path = DEST_DIR / shapes_file
    centers_path = DEST_DIR / centers_file
    
    if shapes_path.exists() and centers_path.exists():
        print(f"  ⏭ Simulation {simulation_num}: Files already exist, skipping...")
        return True
    
    # Transfer shapes file
    shapes_task = transfer_file(source_ep, dest_ep, simulation_num, 'shapes')
    if shapes_task:
        wait_for_transfer(shapes_task, verbose=False)
    
    # Transfer centers file
    centers_task = transfer_file(source_ep, dest_ep, simulation_num, 'centers')
    if centers_task:
        wait_for_transfer(centers_task, verbose=False)
    
    if shapes_task and centers_task:
        print(f"  ✓ Simulation {simulation_num}: Download completed")
        return True
    else:
        print(f"  ❌ Simulation {simulation_num}: Some transfers failed")
        return False


def main():
    """Main function to download simulations 2-100."""
    print("=" * 60)
    print(f"Globus Download Script for Simulations {START_SIM}-{END_SIM}")
    print("=" * 60)
    
    # Check Globus CLI
    if not check_globus_cli():
        sys.exit(1)
    
    # Check authentication
    if not check_authentication():
        print("\nPlease authenticate first:")
        print("  globus login")
        sys.exit(1)
    
    # Find destination endpoint
    dest_endpoint = find_destination_endpoint()
    if not dest_endpoint:
        print("\nError: Could not determine destination endpoint")
        sys.exit(1)
    
    # Ensure destination directory exists
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download simulations
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\nStarting download of simulations {START_SIM} to {END_SIM}...")
    print(f"Total simulations to download: {END_SIM - START_SIM + 1}")
    print(f"Destination: {DEST_DIR}\n")
    
    for sim_num in range(START_SIM, END_SIM + 1):
        try:
            result = download_simulation(SOURCE_ENDPOINT_ID, dest_endpoint, sim_num)
            if result:
                successful += 1
            else:
                failed += 1
            
            # Small delay between simulations to avoid overwhelming the system
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠ Download interrupted by user")
            break
        except Exception as e:
            print(f"  ❌ Simulation {sim_num}: Unexpected error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total processed: {successful + failed}")
    print(f"\nFiles saved to: {DEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
