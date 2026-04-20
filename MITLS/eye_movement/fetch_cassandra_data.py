#!/usr/bin/env python3
"""
Cassandra Data Retrieval Test Script

This script connects to a Cassandra database and retrieves data to save as CSV.

WHAT IS A KEYSPACE?
-------------------
A keyspace in Cassandra is similar to a "database" or "schema" in traditional SQL databases.
It's the top-level container that holds tables (called "column families" in Cassandra).

Think of it like:
- MySQL/PostgreSQL: Database > Tables
- Cassandra: Keyspace > Tables

A Cassandra cluster can have multiple keyspaces, each with its own tables and replication settings.
Example keyspaces might be: "flight_data", "user_data", "logs", etc.
"""

import csv
from pathlib import Path
from datetime import datetime

# Fix for Python 3.12+ compatibility (asyncore removed)
# Import eventlet and monkey patch BEFORE importing cassandra
import eventlet
eventlet.monkey_patch()

# Now we can safely import Cassandra with eventlet support
from cassandra.cluster import Cluster
from cassandra.io.eventletreactor import EventletConnection
from cassandra.auth import PlainTextAuthProvider

# Cassandra connection settings
CASSANDRA_HOST = '127.0.0.1'
CASSANDRA_PORT = 9042

# Output directory
BASE_DIR = Path(__file__).parent.absolute()
OUTPUT_CSV = BASE_DIR / "team-analyzer" / "cassandra_export.csv"


def explore_cassandra_structure():
    """
    Connect to Cassandra and explore available keyspaces and tables.
    This helps you understand what data is available.
    """
    print("=" * 80)
    print("CASSANDRA DATABASE EXPLORATION")
    print("=" * 80)
    
    try:
        # Connect to Cassandra (no authentication for now)
        print(f"\nConnecting to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT}...")
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, connection_class=EventletConnection)
        session = cluster.connect()
        print("✓ Connected successfully!\n")
        
        # List all keyspaces
        print("Available Keyspaces:")
        print("-" * 80)
        keyspaces_query = "SELECT keyspace_name FROM system_schema.keyspaces"
        keyspaces = session.execute(keyspaces_query)
        
        user_keyspaces = []
        for row in keyspaces:
            keyspace_name = row.keyspace_name
            # Filter out system keyspaces
            if not keyspace_name.startswith('system'):
                user_keyspaces.append(keyspace_name)
                print(f"  • {keyspace_name}")
        
        if not user_keyspaces:
            print("  (No user keyspaces found - only system keyspaces exist)")
            print("\n⚠ You may need to create a keyspace and tables first.")
            cluster.shutdown()
            return None
        
        print()
        
        # For each user keyspace, list tables
        for keyspace in user_keyspaces:
            print(f"\nTables in keyspace '{keyspace}':")
            print("-" * 80)
            
            tables_query = f"""
                SELECT table_name 
                FROM system_schema.tables 
                WHERE keyspace_name = '{keyspace}'
            """
            tables = session.execute(tables_query)
            
            table_list = []
            for row in tables:
                table_name = row.table_name
                table_list.append(table_name)
                print(f"  • {table_name}")
            
            if not table_list:
                print("  (No tables found in this keyspace)")
        
        cluster.shutdown()
        return user_keyspaces
        
    except Exception as e:
        print(f"\n❌ Error exploring Cassandra: {e}")
        print("\nPossible issues:")
        print("  1. Cassandra is not running")
        print("  2. Wrong host/port")
        print("  3. Authentication required")
        print("  4. Docker container not accessible")
        return None


def fetch_table_data(keyspace_name, table_name, output_csv_path=None, limit=1000):
    """
    Fetch data from a specific Cassandra table and save as CSV.
    
    Args:
        keyspace_name: The keyspace (database) name
        table_name: The table name
        output_csv_path: Path to save the CSV (optional, uses default if not provided)
        limit: Maximum number of rows to fetch
    
    Returns:
        Path to the generated CSV file, or None if failed
    """
    print("\n" + "=" * 80)
    print(f"FETCHING DATA: {keyspace_name}.{table_name}")
    print("=" * 80)
    
    # Use provided path or default
    csv_path = output_csv_path if output_csv_path else OUTPUT_CSV
    
    try:
        # Connect to Cassandra
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, connection_class=EventletConnection)
        session = cluster.connect()
        
        # Set keyspace
        session.set_keyspace(keyspace_name)
        print(f"✓ Using keyspace: {keyspace_name}\n")
        
        # First, get table schema to understand columns
        print("Table schema:")
        print("-" * 80)
        schema_query = f"""
            SELECT column_name, type 
            FROM system_schema.columns 
            WHERE keyspace_name = '{keyspace_name}' 
            AND table_name = '{table_name}'
        """
        schema = session.execute(schema_query)
        
        columns = []
        for row in schema:
            columns.append(row.column_name)
            print(f"  {row.column_name}: {row.type}")
        
        if not columns:
            print("  (No columns found - table might be empty or doesn't exist)")
            cluster.shutdown()
            return None
        
        print()
        
        # Fetch data from table
        print(f"Fetching up to {limit} rows...")
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        rows = session.execute(query)
        
        # Convert to list to count and iterate
        rows_list = list(rows)
        print(f"✓ Retrieved {len(rows_list)} rows\n")
        
        if not rows_list:
            print("⚠ No data found in table")
            cluster.shutdown()
            return None
        
        # Save to CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to CSV: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Use semicolon delimiter to match your existing CSV format
            writer = csv.writer(csvfile, delimiter=';')
            
            # Write header
            writer.writerow(columns)
            
            # Write data rows
            for row in rows_list:
                # Convert row to list of values in column order
                row_values = [getattr(row, col) for col in columns]
                writer.writerow(row_values)
        
        print(f"✓ CSV file created: {csv_path}")
        print(f"  Rows: {len(rows_list)}")
        print(f"  Columns: {len(columns)}")
        
        cluster.shutdown()
        return str(csv_path)
        
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        return None


def fetch_latest_data_with_timestamp(keyspace_name, table_name, timestamp_column, limit=1000):
    """
    Fetch the most recent data based on a timestamp column.
    This is useful when you want only the latest records.
    
    Args:
        keyspace_name: The keyspace name
        table_name: The table name
        timestamp_column: Name of the column containing timestamps
        limit: Maximum number of rows to fetch
    """
    print("\n" + "=" * 80)
    print(f"FETCHING LATEST DATA: {keyspace_name}.{table_name}")
    print("=" * 80)
    
    try:
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, connection_class=EventletConnection)
        session = cluster.connect()
        session.set_keyspace(keyspace_name)
        
        # Note: ORDER BY in Cassandra only works on clustering columns
        # This might not work depending on your table schema
        # You may need to adjust based on your partition key and clustering columns
        query = f"""
            SELECT * FROM {table_name} 
            ORDER BY {timestamp_column} DESC 
            LIMIT {limit}
        """
        
        print(f"Executing query with ORDER BY {timestamp_column}...")
        rows = session.execute(query)
        
        # Process same as above...
        rows_list = list(rows)
        print(f"✓ Retrieved {len(rows_list)} rows")
        
        # Save to CSV (same logic as fetch_table_data)
        # ... (implement CSV writing here)
        
        cluster.shutdown()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: ORDER BY in Cassandra requires the column to be a clustering key.")
        print("You may need to query all data and sort in Python instead.")


def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("CASSANDRA CSV EXPORT TEST")
    print("=" * 80)
    print(f"\nCassandra endpoint: {CASSANDRA_HOST}:{CASSANDRA_PORT}")
    print(f"Output directory: {OUTPUT_CSV.parent}\n")
    
    # Step 1: Explore database structure
    keyspaces = explore_cassandra_structure()
    
    if not keyspaces:
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("\n1. Make sure Cassandra is running in Docker:")
        print("   docker ps | grep cassandra")
        print("\n2. Connect to Cassandra shell to check data:")
        print("   docker exec -it <container_name> cqlsh")
        print("\n3. Inside cqlsh, run:")
        print("   DESCRIBE KEYSPACES;")
        print("   USE your_keyspace_name;")
        print("   DESCRIBE TABLES;")
        print("   SELECT * FROM your_table LIMIT 10;")
        return
    
    # Step 2: If keyspaces exist, prompt for which one to export
    print("\n" + "=" * 80)
    print("DATA EXPORT")
    print("=" * 80)
    print("\nTo export data, you need to specify:")
    print("  1. Keyspace name (like a database)")
    print("  2. Table name (like a table in SQL)")
    print("\nExample usage in your code:")
    print("  fetch_table_data('flight_data', 'task_events', limit=1000)")
    print("\nAttempting to export data from all tables in 'ingescape' keyspace...")
    print()
    
    # Export data from each table to see what's available
    for table in ['record', 'event', 'blob']:
        print(f"\n{'='*80}")
        print(f"Exporting: ingescape.{table}")
        print('='*80)
        
        # Set output path for each table
        output_path = BASE_DIR / f"cassandra_{table}_export.csv"
        
        fetch_table_data('ingescape', table, output_csv_path=output_path, limit=1000)


if __name__ == "__main__":
    main()
