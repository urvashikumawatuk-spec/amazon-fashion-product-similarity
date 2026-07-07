from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import SimilarProductsRequest

router = APIRouter()


@router.post("/find_similar_products")
def find_similar_products(
    request: Request,
    payload: SimilarProductsRequest,
):
    engine = request.app.state.engine

    try:
        results = engine.find_similar_products(
            product_id=payload.product_id,
            top_k=payload.top_k,
        )

        return {
            "query_product": payload.product_id,
            "top_k": payload.top_k,
            "results": results.to_dict(orient="records"),
        }

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Product not found.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )