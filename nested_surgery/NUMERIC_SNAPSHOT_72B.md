# Can a donor edit distinguish a saved value from a later overwrite?

In Python, assigning a number to a second variable preserves that number when the original variable is subsequently reassigned. The test pairs a function that returns the saved value with a function that returns the current value.

```python
value = 2
saved = value
value = 7
```

In one prompt the function returns `saved + 1`. In the other it returns `value + 1`. The generated panel inserts three to five filler assignments between saving and overwriting the value. For both lexical contexts use two paired variants of the original and replacement numbers. The returned digit is never any of the assignment digits. Changing the first assignment should affect only a read of the saved value. Changing the overwrite should affect only a read of the current value.

Keep the 7B donor, the 72B recipient, the seven-dimensional alignment, the patch boundary after block 17, and the arms from the read-versus-unused study. Use a donor digit from a separate program. Exchange only one state at a time, at the first assignment or at the overwrite. Score the unmodified model and the donor, full-state and equal-norm orthogonal interventions with the same whole-vocabulary first-answer rule. Here an unused edit means the assignment has no effect on the returned variable, even though the saved binding reads the first value earlier in the program.

Qualify independently on 64 groups and require 48 with all four replies correct. Use 256 disjoint test groups, requiring at least 100 native-qualified groups. Require 80% switching in both lexical contexts for both the selected-site donor and full-state exchange, 95% answer preservation under an unused-site donor edit, and a paired donor-over-orthogonal lead of at least 20 points with a positive lower bound. Report the first assignment and overwrite contexts separately. This is a test of saved-value behavior in a frozen model, motivated by the forward reference proposal. It makes no claim about the benefits of a physically fixed connection in a newly trained architecture. [Original comparison](numeric_select_72b_study/README.md)
