from typing import Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue

class MetadataFilterBuilder:
    """to build the reusable Qdrant metadata filters.
    """
    def __init__(self):
        self.conditions = []
    def user(self, user_id: Optional[str]):
        if user_id:
            self.conditions.append(
                FieldCondition(
                    key="uploaded_by",
                    match=MatchValue(value=user_id)
                )
            )
        return self
    
    def tenant(self,tenant_id: Optional[str]):
        if tenant_id:
            self.conditions.append(
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id),
            )
        )
            return self
    
    def extenstion(self, extension: Optional[str]):
        if extension:
            self.conditions.append(
                FieldCondition(
                    key="file_extension",
                    match=MatchValue(value=extension)
                )
            )
        return self
    
    def version(self, version: Optional[str]):
        if version:
            self.conditions.append(
                FieldCondition(
                    key="version",
                    match=MatchValue(value=version)
                )
            )
        return self
    def active_only(self):
        self.conditions.append(
            FieldCondition(
                key="is_active",
                match=MatchValue(value=True)
            )
        )
        return self
    
    def build(self):
        return Filter(
            must = self.conditions
        )