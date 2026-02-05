#!/usr/bin/env python3
"""
Cassandra Database Inspector
Test script to see exactly what's in Cassandra right now.
"""

import argparse
import eventlet
eventlet.monkey_patch()

from cassandra.cluster import Cluster
from cassandra.io.eventletreactor import EventletConnection
from datetime import datetime

CASSANDRA_HOST = '127.0.0.1'
CASSANDRA_PORT = 9042

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def clear_all_tables():
    """Clear all data from all tables in all keyspaces."""
    print_header("CLEARING ALL CASSANDRA TABLES")
    print(f"Connecting to: {CASSANDRA_HOST}:{CASSANDRA_PORT}\n")
    
    try:
        # Connect
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, connection_class=EventletConnection)
        session = cluster.connect()
        print("✓ Connected successfully!\n")
        
        # Get all keyspaces
        keyspaces_query = "SELECT keyspace_name FROM system_schema.keyspaces"
        keyspaces = session.execute(keyspaces_query)
        
        user_keyspaces = []
        for row in keyspaces:
            keyspace_name = row.keyspace_name
            if not keyspace_name.startswith('system'):
                user_keyspaces.append(keyspace_name)
        
        if not user_keyspaces:
            print("No user keyspaces found. Nothing to clear.")
            cluster.shutdown()
            return
        
        print(f"Found {len(user_keyspaces)} user keyspace(s): {', '.join(user_keyspaces)}\n")
        
        # For each keyspace, clear all tables
        total_tables_cleared = 0
        for keyspace in user_keyspaces:
            print(f"\n{'=' * 80}")
            print(f"KEYSPACE: {keyspace}")
            print('=' * 80)
            
            # Get all tables in this keyspace
            tables_query = f"""
                SELECT table_name 
                FROM system_schema.tables 
                WHERE keyspace_name = '{keyspace}'
            """
            tables = session.execute(tables_query)
            table_list = [row.table_name for row in tables]
            
            if not table_list:
                print("  No tables to clear.")
                continue
            
            session.set_keyspace(keyspace)
            
            for table_name in table_list:
                print(f"\nClearing table: {table_name}")
                
                # Get row count before
                count_before = session.execute(f"SELECT COUNT(*) FROM {table_name}").one()[0]
                print(f"  Rows before: {count_before}")
                
                if count_before == 0:
                    print(f"  ✓ Already empty")
                    continue
                
                # Truncate the table
                try:
                    session.execute(f"TRUNCATE {table_name}")
                    
                    # Verify it's empty
                    count_after = session.execute(f"SELECT COUNT(*) FROM {table_name}").one()[0]
                    print(f"  Rows after: {count_after}")
                    
                    if count_after == 0:
                        print(f"  ✓ Successfully cleared {count_before} rows")
                        total_tables_cleared += 1
                    else:
                        print(f"  ⚠ Warning: {count_after} rows still remain")
                        
                except Exception as e:
                    print(f"  ❌ Error truncating table: {e}")
        
        cluster.shutdown()
        
        print_header("CLEAR COMPLETE")
        print(f"\nTotal tables cleared: {total_tables_cleared}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def inspect_cassandra():
    """Connect to Cassandra and inspect all data."""
    
    print_header("CASSANDRA DATABASE INSPECTOR")
    print(f"Connecting to: {CASSANDRA_HOST}:{CASSANDRA_PORT}\n")
    
    try:
        # Connect
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, connection_class=EventletConnection)
        session = cluster.connect()
        print("✓ Connected successfully!\n")
        
        # List keyspaces
        print_header("AVAILABLE KEYSPACES")
        keyspaces_query = "SELECT keyspace_name FROM system_schema.keyspaces"
        keyspaces = session.execute(keyspaces_query)
        
        user_keyspaces = []
        for row in keyspaces:
            keyspace_name = row.keyspace_name
            if not keyspace_name.startswith('system'):
                user_keyspaces.append(keyspace_name)
                print(f"  • {keyspace_name}")
        
        if not user_keyspaces:
            print("  (No user keyspaces found)")
            cluster.shutdown()
            return
        
        # For each keyspace, inspect tables
        for keyspace in user_keyspaces:
            print_header(f"KEYSPACE: {keyspace}")
            
            # List tables
            tables_query = f"""
                SELECT table_name 
                FROM system_schema.tables 
                WHERE keyspace_name = '{keyspace}'
            """
            tables = session.execute(tables_query)
            table_list = [row.table_name for row in tables]
            
            if not table_list:
                print("  (No tables found)")
                continue
            
            print(f"\nTables in '{keyspace}':")
            for table_name in table_list:
                print(f"  • {table_name}")
            
            # Inspect each table
            session.set_keyspace(keyspace)
            
            for table_name in table_list:
                print(f"\n{'-' * 80}")
                print(f"TABLE: {keyspace}.{table_name}")
                print('-' * 80)
                
                # Get table schema
                schema_query = f"""
                    SELECT column_name, type, kind
                    FROM system_schema.columns 
                    WHERE keyspace_name = '{keyspace}' 
                    AND table_name = '{table_name}'
                """
                schema = session.execute(schema_query)
                
                columns = []
                partition_keys = []
                clustering_keys = []
                
                print("\nColumns:")
                for col in schema:
                    columns.append(col.column_name)
                    kind_str = f" [{col.kind}]" if col.kind != 'regular' else ""
                    print(f"  {col.column_name}: {col.type}{kind_str}")
                    
                    if col.kind == 'partition_key':
                        partition_keys.append(col.column_name)
                    elif col.kind == 'clustering':
                        clustering_keys.append(col.column_name)
                
                # Count rows
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                result = session.execute(count_query)
                row_count = result.one()[0]
                print(f"\nTotal rows: {row_count}")
                
                if row_count == 0:
                    print("  (Table is empty)")
                    continue
                
                # Get sample data
                print(f"\nSample data (first 10 rows):")
                sample_query = f"SELECT * FROM {table_name} LIMIT 10"
                samples = session.execute(sample_query)
                
                for i, row in enumerate(samples, 1):
                    print(f"\n  Row {i}:")
                    for col in columns:
                        value = getattr(row, col, None)
                        # Format for readability
                        if isinstance(value, bytes):
                            # Try to decode or show as hex
                            try:
                                value = value.decode('utf-8')
                            except:
                                value = f"<bytes: {value.hex()[:40]}...>"
                        print(f"    {col}: {value}")
                
                # Special analysis for 'event' table
                if table_name == 'event':
                    print(f"\n{'-' * 80}")
                    print("EVENT TABLE ANALYSIS")
                    print('-' * 80)
                    
                    # Count by agent
                    print("\nRows by agent:")
                    agent_query = "SELECT agent, COUNT(*) as count FROM event GROUP BY agent"
                    try:
                        agents = session.execute(agent_query)
                        for agent in agents:
                            print(f"  {agent.agent}: {agent.count} rows")
                    except Exception as e:
                        print(f"  (Cannot group by agent: {e})")
                        # Alternative: count manually
                        all_rows = session.execute(f"SELECT agent FROM event")
                        agent_counts = {}
                        for r in all_rows:
                            agent_counts[r.agent] = agent_counts.get(r.agent, 0) + 1
                        for agent, count in sorted(agent_counts.items()):
                            print(f"  {agent}: {count} rows")
                    
                    # Look for run markers
                    print("\nSearching for run markers:")
                    # Scan all rows (Cassandra doesn't have 'timestamp' column)
                    all_events = session.execute("SELECT agent, source, value, date, time FROM event")
                    found_markers = False
                    for evt in all_events:
                        if evt.agent == 'Cognitive_Model' and evt.source == 'start':
                            value_str = "true" if evt.value == b'\x01' else "false" if evt.value == b'\x00' else str(evt.value)
                            print(f"  Found: Cognitive_Model;start;{value_str} at {evt.date} {evt.time}")
                            found_markers = True
                    if not found_markers:
                        print("  No Cognitive_Model;start markers found")
                    
                    # Look for end markers
                    print("\nSearching for end markers:")
                    all_events = session.execute("SELECT agent, source, date, time FROM event")
                    found_end = False
                    for evt in all_events:
                        if evt.agent == 'TARS Agent' and evt.source == 'end_signal':
                            print(f"  Found: TARS Agent;end_signal at {evt.date} {evt.time}")
                            found_end = True
                    if not found_end:
                        print("  No TARS Agent;end_signal markers found")
                    
                    # Show time range
                    print("\nTime range analysis:")
                    all_events = session.execute("SELECT date, time FROM event")
                    times = [(evt.date, evt.time) for evt in all_events]
                    if times:
                        times.sort()
                        print(f"  Earliest: {times[0][0]} {times[0][1]}")
                        print(f"  Latest: {times[-1][0]} {times[-1][1]}")
        
        cluster.shutdown()
        print_header("INSPECTION COMPLETE")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cassandra Database Inspector - View or clear database contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Inspect database contents (default)
  python test_cassandra_contents.py
  
  # Clear all tables in all keyspaces
  python test_cassandra_contents.py --clear
  python test_cassandra_contents.py -c
        '''
    )
    
    parser.add_argument(
        '-c', '--clear',
        action='store_true',
        help='Clear all data from all tables in all user keyspaces'
    )
    
    args = parser.parse_args()
    
    if args.clear:
        # Ask for confirmation
        print("\n" + "!" * 80)
        print("  WARNING: This will DELETE ALL DATA from ALL TABLES!")
        print("!" * 80)
        response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            clear_all_tables()
        else:
            print("\nOperation cancelled.")
    else:
        inspect_cassandra()
