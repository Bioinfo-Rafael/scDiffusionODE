# Paper and implementation provenance

Checked on **2026-09-12 JST** before implementing the estimator.

The identity was verified as **Dat Nguyen-Cong, Luong Tran, Tung Kieu**, *When
Denoising Hurts: Rethinking the Terminal Step of Diffusion Time Series Forecasters
-- Extended Version*, **arXiv:2608.14067v1**, submitted 2026-08-14.

| Checked source | Finding |
| --- | --- |
| [arXiv abstract](https://arxiv.org/abs/2608.14067) | Matching title, three authors and identifier; no paper-specific source repository linked in the retrieved abstract. Generic arXiv code-finder toggles are not a repository. |
| [Paper HTML](https://arxiv.org/html/2608.14067v1) and [PDF](https://arxiv.org/pdf/2608.14067) | Read Eq. 7, Eq. 8, Algorithm 1 and Appendix D. PDF text searches for `github.com` and `Code` returned no matches. |
| [Aalborg paper record](https://vbn.aau.dk/en/publications/when-denoising-hurts-rethinking-the-terminal-step-of-diffusion-ti/) and [author profile](https://vbn.aau.dk/en/persons/tungkvt/) | Matching research record; no verified implementation located. Searches by Dat Nguyen-Cong's name also did not establish a source repository. |
| GitHub-targeted web searches | Queries `"When Denoising Hurts" site:github.com`, `"2608.14067" github`, and the full terminal-step title did not locate a matching implementation. Direct GitHub search page retrieval was blocked, so it cannot establish completeness. |
| [Papers with Code](https://paperswithcode.com/) | Identifier/domain search found no matching indexed implementation; direct candidate page retrieval failed. |
| [CatalyzeX record](https://www.catalyzex.com/paper/when-denoising-hurts-rethinking-the-terminal) | Displays “Request Code”, not a repository link. Used only as a code-discovery lead, not an algorithm source. |

**Conclusion:** no official source implementation was found in the public sources
checked. This is a bounded search result, not proof that no repository exists.
The GitHub result `eka-care/when-denoising-hurts` concerns medical speech recognition,
not this time-series forecasting paper, and was excluded.

No external source was cloned or copied. Upstream repository, commit SHA, license
and reused upstream functions are therefore `null` / empty in `src/settings.py`.
The implementation derives the estimator from the mathematical definitions:

- Eq. 7: average clean predictions over S trajectories and all gene features.
- Eq. 8: unconstrained independent affine least-squares fits for `t < boundary`
  and `t >= boundary`; minimize summed residual squares. The latter segment
  includes the boundary, even when stored in decreasing diffusion-time order.
- Algorithm 1: obtain one breakpoint per group and take the exact median.
- Appendix D: exclude 10% of snapshot candidates on either side (16 of 20 remain).

Explicit adaptations requested for this repository: unconditional replicate groups
replace conditional pilots; complete all 1000 ancestral updates and store 20
post-update outputs, ending after t=0. This differs from the paper's 20-step DDIM
schedule `{1000,950,...,50}`. No Bernoulli training or retraining is implemented.

Tie breaking uses the first minimum in increasing snapshot order. The paper does
not prescribe a tie rule. Even-group medians may lie between saved snapshots;
we preserve the half-index and interpolated time coordinates without rounding.
