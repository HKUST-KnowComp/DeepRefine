#!/bin/bash

# Define server directories
SERVERS=(
  "../neo4j-server-wiki"
  "../neo4j-server-pes2o"
  "../neo4j-server-cc"
  "../neo4j-server-dulce"
)

for SERVER in "${SERVERS[@]}"; do
  if [ -d "$SERVER" ]; then
    echo "Stopping Neo4j server in $SERVER..."
    "$SERVER/bin/neo4j" stop
  else
    echo "Directory $SERVER does not exist, skipping."
  fi
done

echo "All available Neo4j servers have been stopped." 