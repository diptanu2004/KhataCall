"""The six things the agent can do, and the shape of their arguments.

Two rules decide the design of every schema here.

**Amounts arrive as words, never as numbers** (hard rule 7). `amount_spoken`
is a string: "dedh sau", not 150. The model's job is to notice that an amount
was said and hand over the words; `nlu/numbers.py` decides what they mean. A
model that returns `150` has already done arithmetic we cannot check, and when
it is wrong it is wrong silently.

**The schemas are the prompt.** With native tool calling they go in the
`tools` field; without it they go in the system prompt and Evon returns one
JSON object (CLAUDE.md §7). Either way these descriptions are what the model
reads, so they are written for a reader who has to decide between six similar
options in one shot.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints


def _absent_if_null_word(value: object) -> object:
    """Treat "None", "null", "" for an optional argument as not given.

    Measured, not hypothetical: asked "kya kya khatam hone wala hai", Evon
    proposed list_low_stock with threshold_spoken="None" - the string. Read
    literally that is an unparseable number and the agent would ask
    "kitne se kam?" for no reason.
    """
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "nil", "undefined"):
        return None
    return value


OptionalSpoken = Annotated[str | None, BeforeValidator(_absent_if_null_word)]

__all__ = [
    "TOOLS",
    "TOOL_SCHEMAS",
    "AddCredit",
    "GetBalance",
    "GetStock",
    "ListLowStock",
    "RecordPayment",
    "ToolArgs",
    "UpdateStock",
    "WRITE_TOOLS",
    "args_model_for",
    "openai_schemas",
]

Spoken = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class AddCredit(BaseModel):
    """Record new udhaar against a customer: goods taken now, paid for later."""

    customer_name: Spoken = Field(description="The customer as the shopkeeper said it, e.g. 'Sharma ji'")
    amount_spoken: Spoken = Field(
        description=(
            "The amount exactly as spoken, in words. 'dedh sau', 'do hazaar paanch sau', "
            "'saadhe teen sau'. Never convert it to digits and never do arithmetic."
        )
    )
    note: OptionalSpoken = Field(default=None, max_length=200, description="What it was for, if said")


class RecordPayment(BaseModel):
    """Record money a customer has paid back, reducing what they owe."""

    customer_name: Spoken = Field(description="The customer as the shopkeeper said it")
    amount_spoken: Spoken = Field(
        description="The amount exactly as spoken, in words. Never convert it to digits."
    )


class GetBalance(BaseModel):
    """Report what a customer currently owes."""

    customer_name: Spoken = Field(description="The customer as the shopkeeper said it")


class UpdateStock(BaseModel):
    """Change the quantity of an item: stock arriving, or stock sold."""

    item: Spoken = Field(description="The item, e.g. 'Parle-G', 'chini', 'Maggi'")
    qty_delta_spoken: Spoken = Field(
        description=(
            "How many, exactly as spoken, in words: 'bees', 'das', 'dedh darjan'. "
            "Put a minus in front when stock is going out: 'minus das' for ten sold."
        )
    )


class GetStock(BaseModel):
    """Report how much of an item is left."""

    item: Spoken = Field(description="The item to check")


class ListLowStock(BaseModel):
    """List items that are running out and need reordering."""

    threshold_spoken: OptionalSpoken = Field(
        default=None,
        max_length=60,
        description=(
            "A spoken cut-off only if one was said, e.g. 'paanch' in 'paanch se kam kya hai'. "
            "Leave it out entirely when none was said."
        ),
    )


ToolArgs = AddCredit | RecordPayment | GetBalance | UpdateStock | GetStock | ListLowStock

TOOLS: dict[str, type[BaseModel]] = {
    "add_credit": AddCredit,
    "record_payment": RecordPayment,
    "get_balance": GetBalance,
    "update_stock": UpdateStock,
    "get_stock": GetStock,
    "list_low_stock": ListLowStock,
}

# Tools that change the book. These are the ones that need the shop owner to say
# yes before anything is written (CLAUDE.md §5).
WRITE_TOOLS = frozenset({"add_credit", "record_payment", "update_stock"})

ToolName = Literal["add_credit", "record_payment", "get_balance", "update_stock", "get_stock", "list_low_stock"]


def args_model_for(name: str) -> type[BaseModel] | None:
    return TOOLS.get(name)


def openai_schemas() -> list[dict]:
    """The tool list in OpenAI function-calling form, for `EvonLLM.complete`."""
    schemas = []
    for name, model in TOOLS.items():
        parameters = model.model_json_schema()
        parameters.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (model.__doc__ or "").strip(),
                    "parameters": parameters,
                },
            }
        )
    return schemas


TOOL_SCHEMAS = openai_schemas()
