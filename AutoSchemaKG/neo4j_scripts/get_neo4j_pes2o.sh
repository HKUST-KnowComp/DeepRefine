#!/bin/bash
ulimit -n 65535
NEO4J_VERSION=2025.03.0
export NEO4J_HOME=`realpath ${PWD}/../neo4j-server-pes2o`
export NEO4J_DATA_DIR=`realpath ${NEO4J_HOME}/data`

${NEO4J_HOME}/bin/neo4j stop


rm -rf `realpath ../neo4j-server-pes2o`
wget https://neo4j.com/artifact.php?name=neo4j-community-${NEO4J_VERSION}-unix.tar.gz -O neo4j.tar.gz
tar xf neo4j.tar.gz
mv neo4j-community-${NEO4J_VERSION} ../neo4j-server-pes2o
rm neo4j.tar.gz


rm -rf $NEO4J_DATA_DIR


# Create APOC configuration
APOC_VERSION=2025.03.0
APOC_FILE=apoc-${APOC_VERSION}-core.jar
if [ ! -f ${NEO4J_HOME}/plugins/${APOC_FILE} ]; then
    wget -P ${NEO4J_HOME}/plugins/ https://github.com/neo4j/apoc/releases/download/${APOC_VERSION}/${APOC_FILE}
fi
echo "apoc.export.file.enabled=true" >> ${NEO4J_HOME}/conf/apoc.conf
echo "apoc.import.file.use_neo4j_config=false" >> ${NEO4J_HOME}/conf/apoc.conf

# GDS Plugin Installation
GDS_VERSION=2.16.0
GDS_FILE=neo4j-graph-data-science-${GDS_VERSION}.jar
if [ ! -f ${NEO4J_HOME}/plugins/${GDS_FILE} ]; then
    wget -P ${NEO4J_HOME}/plugins/ https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/neo4j-graph-data-science-${GDS_VERSION}.jar
fi
echo "dbms.security.procedures.unrestricted=apoc.*,n10s.*,gds.*" >> ${NEO4J_HOME}/conf/neo4j.conf
echo "dbms.security.procedures.allowlist=apoc.*,n10s.*,gds.*" >> ${NEO4J_HOME}/conf/neo4j.conf

# RDF Plugin Installation
NEOSEM_VERSION=5.20.0
NEOSEM_FILE=neosemantics-${NEOSEM_VERSION}.jar
if [ ! -f ${NEO4J_HOME}/plugins/${NEOSEM_FILE} ]; then
    wget -P ${NEO4J_HOME}/plugins/ https://github.com/neo4j-labs/neosemantics/releases/download/${NEOSEM_VERSION}/${NEOSEM_FILE}
fi
echo "dbms.unmanaged_extension_classes=n10s.endpoint=/rdf" >> ${NEO4J_HOME}/conf/neo4j.conf

# Change the default port to the unused ports in case there are also other people running Neo4j on the same machine

# sed -i "s/^#\?server.bolt.listen_address=.*$/server.bolt.listen_address=:9687/" ${NEO4J_HOME}/conf/neo4j.conf
# sed -i "s/^#\?server.bolt.advertised_address=.*$/server.bolt.advertised_address=:9687/" ${NEO4J_HOME}/conf/neo4j.conf

# sed -i "s/^#\?server.http.listen_address=.*$/server.http.listen_address=:9474/" ${NEO4J_HOME}/conf/neo4j.conf
# sed -i "s/^#\?server.http.advertised_address=.*$/server.http.advertised_address=:9474/" ${NEO4J_HOME}/conf/neo4j.conf

# sed -i "s/^#\?server.https.listen_address=.*$/server.https.listen_address=:9473/" ${NEO4J_HOME}/conf/neo4j.conf
# sed -i "s/^#\?server.https.advertised_address=.*$/server.https.advertised_address=:9473/" ${NEO4J_HOME}/conf/neo4j.conf


# Set initial password using the corrected command
echo "Setting initial password..."
${NEO4J_HOME}/bin/neo4j-admin dbms set-initial-password admin2024 || echo "Password already set or error."

# Start Neo4j
# ${NEO4J_HOME}/bin/neo4j start
# sleep 10

# # Restart Neo4j
# ${NEO4J_HOME}/bin/neo4j restart
# sleep 10

# Creating index and initializing RDF configuration
# ${NEO4J_HOME}/bin/cypher-shell -u neo4j -p admin2024 "CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;"
# ${NEO4J_HOME}/bin/cypher-shell -u neo4j -p admin2024 'CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;'
# ${NEO4J_HOME}/bin/cypher-shell -u neo4j -p admin2024 'call n10s.graphconfig.init({ handleMultival: "OVERWRITE", handleVocabUris: "SHORTEN", keepLangTag: false, handleRDFTypes: "NODES" })'

# Output Neo4j log
# tail -n 12 $NEO4J_HOME/logs/neo4j.log