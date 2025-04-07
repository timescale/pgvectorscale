import re
import csv

def extract_first_number(text):
    # Extract the first number from a string, handling both integer and float values
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return match.group(1)
    return ''

def parse_log(log_file, output_file):
    # Define the header for the CSV
    header = ['TABLE', 'RUN', 'NUM_LABELS', 'MIN', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99', 'MAX', 'MEAN', 'QPS']
    
    # Open the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Read the log file
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Split the content into individual runs
        sections = content.split('----------------------------------------')
        
        current_run = None
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # Check if this section contains a run header
            match = re.match(r'(.+?)\s+run\s+(\d+)\s+num_labels\s+(\d+)', lines[0])
            if match:
                # If we have a previous run with stats, write it
                if current_run and 'stats' in current_run:
                    writer.writerow([
                        current_run['table'],
                        current_run['run_num'],
                        current_run['num_labels'],
                        current_run['stats'].get('MIN', ''),
                        current_run['stats'].get('P25', ''),
                        current_run['stats'].get('P50', ''),
                        current_run['stats'].get('P75', ''),
                        current_run['stats'].get('P90', ''),
                        current_run['stats'].get('P95', ''),
                        current_run['stats'].get('P99', ''),
                        current_run['stats'].get('MAX', ''),
                        current_run['stats'].get('MEAN', ''),
                        current_run['stats'].get('QPS', '')
                    ])
                
                # Start a new run
                current_run = {
                    'table': match.group(1),
                    'run_num': match.group(2),
                    'num_labels': match.group(3),
                    'stats': {}
                }
            
            # Check if this section contains statistics
            if current_run and 'Query Time Statistics' in section:
                stats = {}
                in_stats = False
                for line in lines:
                    if 'Query Time Statistics' in line:
                        in_stats = True
                        continue
                    if in_stats and ':' in line:
                        key, value = line.split(':')
                        key = key.strip()
                        value = value.strip()
                        if key == 'p50/Median':
                            key = 'P50'
                        elif key == 'Queries':
                            continue
                        elif key == 'Total':
                            continue
                        stats[key.upper()] = extract_first_number(value)
                    elif 'QPS:' in line:
                        stats['QPS'] = extract_first_number(line.split(':')[1].strip())
                
                if stats:
                    current_run['stats'] = stats
        
        # Write the last run if it exists
        if current_run and 'stats' in current_run:
            writer.writerow([
                current_run['table'],
                current_run['run_num'],
                current_run['num_labels'],
                current_run['stats'].get('MIN', ''),
                current_run['stats'].get('P25', ''),
                current_run['stats'].get('P50', ''),
                current_run['stats'].get('P75', ''),
                current_run['stats'].get('P90', ''),
                current_run['stats'].get('P95', ''),
                current_run['stats'].get('P99', ''),
                current_run['stats'].get('MAX', ''),
                current_run['stats'].get('MEAN', ''),
                current_run['stats'].get('QPS', '')
            ])

if __name__ == '__main__':
    parse_log('run.log', 'results.csv') 