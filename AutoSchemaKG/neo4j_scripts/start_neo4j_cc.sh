#!/bin/bash
# ulimit -n 65535
NEO4J_VERSION=5.24.1


export NEO4J_HOME=`realpath ${PWD}/../neo4j-server-cc`
export NEO4J_DATA_DIR=`realpath ${NEO4J_HOME}/data`
# export CLASSPATH_PREFIX=`realpath ${NEO4J_HOME}/lib/dozerdb-plugin-5.24.2.1.jar`
# export CLASSPATH_PREFIX=$NEO4J_HOME/lib/dozerdb-plugin-5.24.2.1-alpha.jar

# Get the database name from the first argument, default to "movie" if not provided
DATABASE_NAME=${1:-cc-csv-json-text}

# Modify the Neo4j configuration
sed -i "s/^#\?initial.dbms.default_database=.*/initial.dbms.default_database=${DATABASE_NAME}/" ${NEO4J_HOME}/conf/neo4j.conf

# Start Neo4j
${NEO4J_HOME}/bin/neo4j stop
${NEO4J_HOME}/bin/neo4j start --verbose
sleep 10

# Creating index and initializing RDF configuration
${NEO4J_HOME}/bin/cypher-shell -u neo4j -p admin2024 'CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;' -a bolt://localhost:8013
${NEO4J_HOME}/bin/cypher-shell -u neo4j -p admin2024 'call n10s.graphconfig.init({ handleMultival: "OVERWRITE", handleVocabUris: "SHORTEN", keepLangTag: false, handleRDFTypes: "NODES" })' -a bolt://localhost:8013






# Output Neo4j log
tail -n 12 $NEO4J_HOME/logs/neo4j.log