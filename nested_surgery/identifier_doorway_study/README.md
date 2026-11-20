# An assignment sign did not relay a declaration edit through a seven-block wall

On 256 new two-definition Python files, an unrestricted declaration edit reversed 242 reads of its class and zero reads of the other class. The same edit reversed zero reads through either of two single-token routes. The assignment sign immediately after the edited declaration changed by a median Euclidean norm of 11.91 at release. Its effect on the final score was small.

## The imposed wall

The frozen Qwen2.5-14B base checkpoint at revision `97e1e76335b7017d8f67c08a19d103c0504298c9` receives the same rank-25 edit to the earlier declaration as in the previous two-class study. The other class and both test queries remain in the file. At the exit of each of seven blocks after block 10, the wall restores every state to the untouched run except the source and one relay. At the end of the seventh block it restores the source too. Only the relay differs in the input to block 18. We test the assignment sign after the first declaration and the assignment sign after the second. Both occur after the edited token, and both remain visible under ordinary causal attention. [Registered protocol](../IDENTIFIER_DOORWAY.md) · [Implementation](../identifier_doorway.py)

| How the edit can travel | Reversed reads of edited class | Reversed reads of other class | Original-letter margin on edited class |
|---|---:|---:|---:|
| Native baseline | — | — | +12.64 |
| Unrestricted | 242 / 256 | 0 / 256 | −4.64 |
| Through assignment sign in the same class | 0 / 256 | 0 / 256 | +12.63 |
| Through assignment sign in the other class | 0 / 256 | 0 / 256 | +12.64 |

The unrestricted edit also reduced the other class's mean margin from +12.60 to +5.91 despite changing no paired winners. The two relay arms left that margin near +12.60. The second relay's median displacement was 1.13. These measurements distinguish carrying an altered state from carrying enough task-relevant information to flip the answer. They test these relay tokens and this window. A result with a different source, receiver or layer range can differ. [Raw execution and margins](results) · [Independent audit](verify.py) · [Recomputed effects](audit.json) · [Query-token follow-up](../IDENTIFIER_QUERY_RELAY.md)

Run `uv run --with numpy python nested_surgery/identifier_doorway_study/verify.py` from the repository root. The pinned run took two minutes and 29 seconds on the owner's Apple M5 Ultra. The owner proposed testing walls that prevent communication between selected parts of a network. OpenAI Codex implemented this intervention, operated the run, checked the results and assisted with writing.
