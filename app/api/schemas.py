from pydantic import BaseModel, Field


class SimilarProductsRequest(BaseModel):
    """
    Request schema for product similarity search.
    """

    product_id: str = Field(
        ...,
        description="Unique product identifier",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of similar products",
    )