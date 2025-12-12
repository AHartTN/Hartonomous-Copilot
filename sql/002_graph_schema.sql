-- Relations table: The actual neural network structure
CREATE TABLE IF NOT EXISTS relations (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT NOT NULL REFERENCES atoms(id),
    child_id BIGINT NOT NULL REFERENCES atoms(id),
    relation_type TEXT NOT NULL,
    weight_atom_id BIGINT REFERENCES atoms(id),
    position INT,
    layer_name TEXT,
    meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_relations_parent ON relations(parent_id);
CREATE INDEX IF NOT EXISTS idx_relations_child ON relations(child_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_flow ON relations(parent_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_layer ON relations(layer_name) WHERE layer_name IS NOT NULL;

-- View for easier querying
CREATE OR REPLACE VIEW token_embeddings AS
SELECT 
    p.id as token_id,
    p.raw_value as token,
    r.position as dim,
    c.raw_value as embedding_value,
    r.layer_name
FROM relations r
JOIN atoms p ON r.parent_id = p.id
JOIN atoms c ON r.child_id = c.id
WHERE r.relation_type = 'embedding';

-- View for attention weights
CREATE OR REPLACE VIEW attention_graph AS
SELECT 
    p.id as from_token,
    c.id as to_token,
    w.raw_value as weight_value,
    r.layer_name,
    r.position
FROM relations r
JOIN atoms p ON r.parent_id = p.id
JOIN atoms c ON r.child_id = c.id
LEFT JOIN atoms w ON r.weight_atom_id = w.id
WHERE r.relation_type LIKE 'attention%';
