# ASD-STE100 Simplified Technical English: the standard for ipm-lstm-attn

The README of ipm-lstm-attn and this file obey these rules. Section 3 gives the project vocabulary: the technical names and the technical verbs of ipm-lstm-attn.

## 1. Rules for text

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the ipm-lstm-attn documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **instance** | One optimisation problem: shared matrices and one right-hand side `b`. | problem (for one item), sample, example |
| **family** | The type of instance: `qp`, `qcqp` or `nonconvex`. | class, problem type |
| **dataset** | All instances that one call of `generate` makes, with one manifest. | corpus, data file |
| **manifest** | The `.json` file next to a dataset. It holds the sizes, the seed and the SHA-256 value. | metadata file, header |
| **split** | The disjoint train, validation and test index sets of a dataset. | fold, partition |
| **KKT system** | The equations `F(y) = 0` of the barrier problem, with the Jacobian `J`. | optimality system, Newton matrix |
| **iterate** | The vector `y = [x, eta, s, lam]` at one IPM step. | point (alone), state, solution (before convergence) |
| **Newton system** | The linear system `J dy = -F` at one iterate. | linear system (alone), step equation |
| **Newton solver** | A component that returns `dy` for a Newton system: `direct`, `cg<k>` or `learned:<variant>`. | linear solver, optimizer, inner solver |
| **learned solver** | A Newton solver that is a trained neural network. | learned optimizer, LSTM solver, model (alone) |
| **variant** | One learned solver design from the registry: `lstm1`, `lstm2`, `lstm2_attn`, `lstm2_mask`, `lstm2_gnn`. | architecture, version, flavour |
| **mixing block** | The attention or message-passing layer that lets tokens exchange information. | attention layer (for both), encoder |
| **token** | One entry of `dy` with its two features (value and gradient). | variable (for an entry of `dy`), node |
| **structure mask** | The pattern of `J^T J`: tokens that share a row of `J`. | sparsity mask, graph mask |
| **outer step** | One IPM iteration: one Newton system and one step. | outer iteration, IPM iteration |
| **inner step** | One update of the learned solver inside one outer step. | inner iteration, unroll step |
| **outer budget** | The fixed number of outer steps of the approximate IPM (`outer_iters`). | horizon, max iterations |
| **reference solve** | The exact IPM run to `reference_tol` that gives `f*`. | ground truth, optimal solve |
| **back-end** | The solver for the warm-start comparison: `exact_ipm` or `ipopt`. | backend solver, engine |
| **cold start** | A back-end solve from the default initial iterate. | cold solve, default start |
| **warm start** | A back-end solve from the iterate of the approximate IPM. | warm solve, hot start |
| **KKT error** | The largest of the dual residual, the two primal residuals and the mean complementarity. | KKT residual (as a single number), optimality gap |
| **objective gap** | `abs(f - f*) / max(1, abs(f*))`. | relative error, optimality gap |
| **ablation** | Training and evaluation of every variant with every seed on the same split. | study, sweep |
| **paired difference** | The mean over test instances of (variant minus baseline), with a bootstrap interval. | delta, improvement |
| **timing protocol** | Warm-up calls, repeats and separate per-instance and batched rows. | benchmark (for the rules), profiling |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **generate** | Make a dataset from a seed. |
| **load** | Read a dataset and check its schema and its SHA-256 value. |
| **train** | Fit the weights of one variant with one seed inside the IPM loop. |
| **evaluate** | Run the approximate IPM, measure the iterate, then do the cold start and the warm start. |
| **precondition** | Scale each row of a Newton system to unit length. |
| **warm-start** | Start a back-end solve from a given iterate. |
| **ablate** | Do an ablation. |
| **measure** | Run the timing protocol on one call. |
