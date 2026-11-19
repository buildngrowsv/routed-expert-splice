"""Native fill-in-the-middle Python repair, with a conventional prompt sensitivity check."""
from problems import SIZES, problem, seed_for
import random

OP_TEXT = (
    ("Retain the flags present in both operands.", "Keep only bits enabled in both patterns.",
     "Clear every bit that either operand leaves unset.", "Find the overlap between the two bit masks."),
    ("Enable every flag present in either operand.", "Merge the two sets of enabled bits.",
     "Keep a bit when at least one operand sets it.", "Add the requested flags without clearing existing flags."),
    ("Toggle the selected flags in the adjusted value.", "Flip each bit requested by the second operand.",
     "Keep only bits on which the two operands disagree.", "Bits set in both operands cancel each other."),
    ("Move the adjusted value left by the requested number of positions.", "Shift the bits toward higher positions.",
     "Scale the adjusted value by two to the power of the second operand.", "Pad the value on the right with the requested number of zero bits."),
)


def infill(split, plain=False):
    rows = []
    for i in range(SIZES[split]):
        row = problem("bits","fim-"+split,i)
        rng = random.Random(seed_for(row["id"]+" description"))
        description = rng.choice(OP_TEXT[row["gold"]])
        head = "# "+description+"\n"+row["head"]
        assertions = "\n".join(f"assert update({x}, {y}) == {out}" for (x,y),out in row["examples"])
        suffix = row["tail"]+"\n\n"+assertions+"\n"
        if plain:
            prompt = "# Fill the blank with the missing Python operator.\n"+head+" ___"+suffix+"# Missing operator:"
        else:
            prompt = "<|fim_prefix|>"+head+"<|fim_suffix|>"+suffix+"<|fim_middle|>"
        row.update(prompt=prompt, description=description)
        rows.append(row)
    return rows
