#!/bin/bash

echo "================================================================================"
echo "  HARTONOMOUS-COPILOT: SQL-BASED LLM INFERENCE"
echo "  Live Demonstration - December 12, 2025"
echo "================================================================================"
echo ""

echo "[1] Database Statistics"
echo "----------------------------------------"
sudo -u postgres psql -d hartonomous -t -c "
SELECT 
    'Atoms: ' || COUNT(*)::text as stat
FROM atoms
UNION ALL
SELECT 
    'Connections: ' || COUNT(*)::text
FROM connections
UNION ALL
SELECT
    'Unique geometries: ' || COUNT(DISTINCT ST_GeometryType(geom))::text
FROM atoms;"

echo ""
echo "[2] Sample Atoms (First 10)"
echo "----------------------------------------"
sudo -u postgres psql -d hartonomous -c "
SELECT 
    id,
    encode(raw_value, 'escape')::text as value,
    substring(ST_AsText(geom), 1, 60) || '...' as geometry
FROM atoms
WHERE raw_value IS NOT NULL AND length(raw_value) < 10
LIMIT 10;"

echo ""
echo "[3] Sample Connections"
echo "----------------------------------------"
sudo -u postgres psql -d hartonomous -c "
SELECT 
    encode(a1.raw_value, 'escape')::text as from_token,
    ' → ' as arrow,
    encode(a2.raw_value, 'escape')::text as to_token,
    substring(encode(a3.raw_value, 'escape')::text, 1, 10) as weight
FROM connections c
JOIN atoms a1 ON a1.id = c.from_id
JOIN atoms a2 ON a2.id = c.to_id
LEFT JOIN atoms a3 ON a3.id = c.weight_id
LIMIT 10;"

echo ""
echo "[4] Live Inference Test: 'Hello'"
echo "----------------------------------------"
sudo -u postgres psql -d hartonomous -c "
WITH prompt AS (
    SELECT id FROM atoms WHERE raw_value = 'Hello'
)
SELECT 
    encode(a.raw_value, 'escape')::text as predicted_next_token,
    c.weight_id
FROM prompt p
JOIN connections c ON c.from_id = p.id
JOIN atoms a ON a.id = c.to_id
ORDER BY c.weight_id DESC
LIMIT 5;"

echo ""
echo "[5] Live Inference Test: 'The'"
echo "----------------------------------------"
sudo -u postgres psql -d hartonomous -c "
WITH prompt AS (
    SELECT id FROM atoms WHERE raw_value = 'The'
)
SELECT 
    encode(a.raw_value, 'escape')::text as predicted_next_token,
    c.weight_id
FROM prompt p
JOIN connections c ON c.from_id = p.id
JOIN atoms a ON a.id = c.to_id
ORDER BY c.weight_id DESC
LIMIT 5;"

echo ""
echo "================================================================================"
echo "  ✓ Demonstration Complete"
echo "================================================================================"
