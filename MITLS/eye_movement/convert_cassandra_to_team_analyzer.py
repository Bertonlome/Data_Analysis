#!/usr/bin/env python3
"""
Convert Cassandra event export to team-analyzer compatible format.

This script transforms the programmatically retrieved Cassandra CSV data
into the format expected by team-analyzer.py (matching manually exported format).
"""

import csv
import struct
from datetime import datetime
from pathlib import Path


def decode_blob_value(blob_str, value_type):
    """
    Decode Cassandra blob value based on type.
    
    Blob values are stored as Python byte strings like: b'\\x9a\\x99...'
    Need to decode based on the event type.
    """
    if not blob_str or blob_str == '':
        return ''
    
    # Remove b' prefix and ' suffix if present
    if blob_str.startswith("b'") and blob_str.endswith("'"):
        blob_str = blob_str[2:-1]
    
    # Handle different value types
    try:
        # Type 2 = DOUBLE
        if value_type == '2':
            # Decode hex escape sequences to bytes
            blob_bytes = blob_str.encode('utf-8').decode('unicode_escape').encode('latin1')
            if len(blob_bytes) == 8:
                double_val = struct.unpack('d', blob_bytes)[0]
                return f"{double_val:.6f}"
        
        # Type 3 = INTEGER
        elif value_type == '3':
            blob_bytes = blob_str.encode('utf-8').decode('unicode_escape').encode('latin1')
            if len(blob_bytes) == 4:
                int_val = struct.unpack('i', blob_bytes)[0]
                return str(int_val)
        
        # Type 1 or 4 = BOOL
        elif value_type in ('1', '4'):
            blob_bytes = blob_str.encode('utf-8').decode('unicode_escape').encode('latin1')
            if len(blob_bytes) > 0:
                bool_val = struct.unpack('?', blob_bytes[:1])[0]
                return 'true' if bool_val else 'false'
        
        # Type 9 = STRING or other text types (14=service, 15=entered, 17=definition, etc.)
        # For strings and JSON, just clean up the encoding
        else:
            # Try to decode as UTF-8 string
            try:
                if '\\x' in blob_str:
                    # Has hex escapes - decode them
                    blob_bytes = blob_str.encode('utf-8').decode('unicode_escape').encode('latin1')
                    decoded = blob_bytes.decode('utf-8', errors='ignore')
                else:
                    # Plain string - just remove escapes
                    decoded = blob_str.replace('\\"', '"').replace('\\n', '\n')
                
                return decoded
            except:
                # If all else fails, return as-is
                return blob_str
    
    except Exception as e:
        # If decoding fails, return the original string
        print(f"Warning: Could not decode blob for type {value_type}: {e}")
        return blob_str
    
    return blob_str

def convert_cassandra_export_to_team_analyzer_format(
    cassandra_csv_path,
    output_csv_path
):
    """
    Convert Cassandra event export format to team-analyzer format.
    
    Cassandra format:
        agent;date;id;igs_timestamp;record_id;source;time;type;value
        TARS Agent;2026-02-03;uuid;timestamp;record_id;current_state;02:44:10;type;b'{"data":"..."}'
    
    Team-analyzer format:
        uuid;timestamp;agent;source;type;igs_timestamp;value
        uuid;1770104650.613;TARS Agent;current_state;type;;{"data":"..."}
    """
    print(f"Converting: {cassandra_csv_path}")
    print(f"Output: {output_csv_path}\n")
    
    converted_rows = []
    
    with open(cassandra_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        for row in reader:
            # Convert date + time to Unix timestamp
            date_str = row['date']  # '2026-02-03'
            time_str = row['time']  # '02:44:10.613303000'
            
            # Truncate time to microseconds (6 digits) if it has nanoseconds
            if '.' in time_str:
                time_parts = time_str.split('.')
                # Keep only first 6 digits of fractional seconds (microseconds)
                time_str = f"{time_parts[0]}.{time_parts[1][:6]}"
            
            # Parse datetime
            datetime_str = f"{date_str} {time_str}"
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
            unix_timestamp = dt.timestamp()
            
            # Decode blob value based on type
            value = decode_blob_value(row['value'], row['type'])
            
            # Create converted row in team-analyzer format
            converted_row = {
                'uuid': row['id'],
                'timestamp': f"{unix_timestamp:.3f}",
                'agent': row['agent'],
                'source': row['source'],
                'type': row['type'],
                'igs_timestamp': row.get('igs_timestamp', ''),
                'value': value
            }
            
            converted_rows.append(converted_row)
    
    # Write converted data
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['uuid', 'timestamp', 'agent', 'source', 'type', 'igs_timestamp', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        writer.writerows(converted_rows)
    
    print(f"✓ Converted {len(converted_rows)} rows")
    print(f"✓ Output saved to: {output_csv_path}")
    
    return output_csv_path


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.absolute()
    
    cassandra_export = BASE_DIR / "cassandra_event_export.csv"
    team_analyzer_format = BASE_DIR / "team-analyzer" / "cassandra_converted.csv"
    
    if cassandra_export.exists():
        convert_cassandra_export_to_team_analyzer_format(
            cassandra_export,
            team_analyzer_format
        )
    else:
        print(f"❌ Cassandra export not found: {cassandra_export}")
        print("Run fetch_cassandra_data.py first to export from Cassandra")
