#!/bin/bash

# Function to kill process on a specific port
kill_process_on_port() {
    local port=$1
    echo "Checking for process on port $port..."
    
    # Find PID of process using the port
    pid=$(lsof -i :$port -t)
    
    if [ -z "$pid" ]; then
        echo "No process found running on port $port"
    else
        echo "Found process (PID: $pid) on port $port. Killing it..."
        kill -9 $pid
        echo "Process on port $port has been terminated."
    fi
}

# Kill processes on specified ports
kill_process_on_port 8000
kill_process_on_port 8001

echo "Done!"