from dataclasses import dataclass
from typing import Optional

@dataclass(slots = True)
class Tenant:
    tenant_id: str
    tenant_name: str
    owner_id: str
    is_active: bool = True
    
class TenantManager:
    """basic multi-agent support for nexusmind
    """
    @staticmethod
    def attach_tenant_metadata(
        chunk: dict,
        tenant: Tenant,
    )-> dict:
        chunk["tenant_id"] = tenant.tenant_id
        chunk["tenant_name"] = tenant.tenant_name
        chunk["owner_id"] = tenant.owner_id
        return chunk
    @staticmethod
    def validate_access(
        chunk: dict,
        tenant_id:str,
    )-> bool:
        return chunk.get("tenant_id") == tenant_id