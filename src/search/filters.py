"""OpenAI-powered filter extraction and normalization."""

from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.database.manager import Manager


class Filters(BaseModel):
    """Structured filters for fashion search."""

    style_query: str = Field(
        None,
        description="The style query with pattern / material / style / texture / silhouette tokens",
    )
    brand: Optional[str] = Field(None, description="The brand of the piece of clothing")
    category: Optional[str] = Field(
        None, description="The category of the piece of clothing"
    )
    color: Optional[str] = Field(None, description="The color of the piece of clothing")
    clean_query: str = Field(
        None, description="The clean query without price filters and ranges"
    )
    min_price: Optional[float] = Field(
        None, description="The minimum price of the piece of clothing"
    )
    max_price: Optional[float] = Field(
        None, description="The maximum price of the piece of clothing"
    )


class Extractor:

    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def __call__(self, text: str) -> Filters:
        """Extract filters from user input."""
        return self.client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": text},
            ],
            text_format=Filters,
            temperature=0.0,
            top_p=0,
        ).output_parsed

    def _get_system_prompt(self):
        """Parse the system prompt for filter extraction."""

        with Manager() as db:
            db.cursor.execute("SELECT DISTINCT category FROM item.attributes")
            categories = db.cursor.fetchall()
            db.cursor.execute("SELECT DISTINCT target_color FROM item.colors")
            colors = db.cursor.fetchall()
            db.cursor.execute("SELECT DISTINCT brand FROM item.attributes")
            brands = db.cursor.fetchall()

        return f"""
            You are a filter extraction and normalization specialist for fashion search.
            Extract structured filters from user messages and MATCH them to available database values.

            Available filter values in database:
            **Available categories:**
            {categories}
            **Available colors:**
            {colors}
            **Available brands:**
            {brands}

            Filter types to extract and normalize (all values are case insensitive):
            **category**: extract category name and match to available categories list above
            - Handle plurals: "shoe" -> "shoes"
            - Must match exactly to a value in the available categories list
            - If no good match found, return None

            **brand**: extract brand name and match to available brands list above
            Match priority (in order):
                - Exact match: "nike" -> "Nike" or "nike"
            - Always put extractions to lower case before matching
            - Prefer shorter brand names (main brand over sub-brand)
            - Must match to a value in the available brands list
            - If no good match found, return None

            **color**: extract color name and match to available colors list above
            - Handle degree modifiers: "reddish" -> "red"
            - Prefer dominant color: "reddish orange" -> "orange"
            - Must match exactly to a value in the available colors list
            - If no good match found, return None

            **price**: extract price ranges, no normalization needed, return numeric values
            - Currency is always GPB. "Pounds" can be ignored.
            - "Under 50" -> max_price: 50
            - "Over 100" -> min_price: 100
            - "Between 50 and 100" -> min_price: 50, max_price: 100
            - "50 to 100" -> min_price: 50, max_price: 100

            **clean_query**: original query without price filters and ranges.
            - Remove price filter requirements from the query
                - "Velvet pants by Ralph Lauren over 100" -> "Velvet pants by Ralph Lauren"
                - "Nike shoes under 50" -> "Nike shoes"
            - If no price filter requirements are present, return the original query

            **style_query**: remove price, category, and brand tokens from the query
            - Examples:
                - "Funky pants" -> "funky"
                - "Leather striped jacket" -> "leather striped"
                - "Short velvet dress by Ralph Lauren" -> "short velvet"
                - "Pink short-sleeved blouse by DKNY over 100" -> "short-sleeved"
                - "Off-shoulder blouse" -> "off-shoulder"
                - "Emerald floral mini dress" -> "floral mini"
                - "Denim pants with a zipper" -> "denim with a zipper"
                - "Striped pants by Tommy Hilfiger" -> "striped"
                - "Dress with polka dots" -> "polka dots"
    
            **GENERAL RULES**:
            - Match extracted values to available values list (case-insensitive, handle variations)
            - Return null if no good match found in available values
            - **IMPORTANT**: always return the clean_query
            - **IMPORTANT**: always return the style_query
        """
