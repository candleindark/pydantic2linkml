# BICAN already has linkml here:
#   https://github.com/brain-bican/models/tree/main/linkml-schema
# Biolink also has linkml:
#   https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml
# openminds is JSON: https://github.com/openMetadataInitiative/openMINDS_core/tree/v4
# ATOM: https://bioportal.bioontology.org/ontologies/ATOM
# ATOM: https://github.com/SciCrunch/NIF-Ontology/blob/atlas/ttl/atom.ttl
# ATOM: https://www.nature.com/articles/s41597-023-02389-4

if __name__ == "__main__":
    import sys

    from pydantic2linkml.cli import app

    sys.exit(app())
