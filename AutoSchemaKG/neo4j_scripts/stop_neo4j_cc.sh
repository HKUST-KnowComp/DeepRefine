#!/bin/bash

SERVER="../neo4j-server-cc"

if [ -d "$SERVER" ]; then
  echo "Stopping Neo4j server in $SERVER..."
  "$SERVER/bin/neo4j" stop
else
  echo "Directory $SERVER does not exist, skipping."
fi 