# The Score Function, Told as a Story
### A step-by-step build-up to Equations 2.1 – 2.7 of *A Survey on Diffusion Models for Inverse Problems* (Daras et al., 2024)

You already know the forward diffusion process: take a clean sample, keep adding Gaussian noise, and eventually it becomes pure noise. What you are missing is the *one object* that makes the reverse direction possible at all: the **score**. Once the score is clear, Equations 2.1–2.7 stop being formulas and become sentences. So the story starts before the diffusion and only reaches the paper's equations in Chapter 4.

Notation used throughout (matches the paper): $\boldsymbol{x} \in \mathbb{R}^n$ is a signal (e.g. an image flattened into a vector), $p_0$ is the unknown data distribution, $p_t$ is the distribution of noisy signals at noise level/time $t$, $\boldsymbol{Z} \sim \mathcal{N}(\boldsymbol{0}, I_n)$ is standard Gaussian noise, and $\nabla_{\boldsymbol{x}}$ is the gradient with respect to the signal (not with respect to network weights — this trips up many people).

---

## Chapter 1 — What is a score, really?

### 1.1 The definition

For any probability density $p(\boldsymbol{x})$, the **score** is

$$
\boxed{\;\boldsymbol{s}(\boldsymbol{x}) \;=\; \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x})\;}
$$

It is a **vector field**: at every point $\boldsymbol{x}$ of the space, it gives you an $n$-dimensional arrow. The arrow points in the direction in which $\log p$ increases fastest — i.e. **"which way is more probable from here?"** — and its length says how steeply probability rises in that direction.

Think of $p$ as a landscape of hills (high density = data-like signals, e.g. natural images) and valleys (low density = garbage). Standing at a point, the score is the compass needle that points uphill. That is the whole intuition. Everything else is bookkeeping.

### 1.2 Why the log? (This is not cosmetic)

Two reasons, and both matter for diffusion models.

**Reason 1: the normalizing constant disappears.** Almost every interesting density is only known up to a constant: $p(\boldsymbol{x}) = \tilde{p}(\boldsymbol{x}) / Z$, where $Z = \int \tilde{p}(\boldsymbol{x})\,d\boldsymbol{x}$ is an impossible-to-compute integral over all of $\mathbb{R}^n$. Then

$$
\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \big(\log \tilde{p}(\boldsymbol{x}) - \log Z\big) = \nabla_{\boldsymbol{x}} \log \tilde{p}(\boldsymbol{x}).
$$

The constant is killed by the gradient. So **you can know the score of a distribution without ever knowing its normalization**. This is exactly why Bayes' rule becomes usable in Eq. 1.2 of the paper: $p(\boldsymbol{x}|\boldsymbol{y}) = p(\boldsymbol{x})p(\boldsymbol{y}|\boldsymbol{x})/p(\boldsymbol{y})$, and the nasty denominator $p(\boldsymbol{y})$ vanishes when you take $\nabla_{\boldsymbol{x}}\log$:

$$
\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}|\boldsymbol{y}) = \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}) + \nabla_{\boldsymbol{x}} \log p(\boldsymbol{y}|\boldsymbol{x}). \tag{1.2}
$$

**Reason 2: the score is scale-free.** $\nabla p$ is tiny wherever $p$ is tiny (which is almost everywhere in high dimensions), so it carries no usable directional information there. $\nabla \log p = \nabla p / p$ divides out the magnitude of $p$ and keeps only the *direction and relative steepness*. A compass needle that is equally readable in the valley and on the hill.

### 1.3 The one example you must be able to do by hand

Let $p(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \sigma^2 I)$. Then

$$
\log p(\boldsymbol{x}) = -\frac{\|\boldsymbol{x}-\boldsymbol{\mu}\|^2}{2\sigma^2} + \text{const}
\quad\Longrightarrow\quad
\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}) = -\frac{\boldsymbol{x}-\boldsymbol{\mu}}{\sigma^2} = \frac{\boldsymbol{\mu}-\boldsymbol{x}}{\sigma^2}.
$$

Read it: the score **points from where you are toward the mean**, with length proportional to how far away you are and inversely proportional to the variance. A narrow Gaussian (small $\sigma$) pulls hard; a wide one pulls gently. Keep this picture — it is going to reappear as "the score points toward the denoised image".

### 1.4 What is the score *for*? Langevin dynamics

Suppose someone hands you the score $\boldsymbol{s}(\boldsymbol{x})$ of a distribution but not the distribution itself. Can you draw samples? Yes — with **Langevin dynamics**:

$$
\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \epsilon\, \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}_k) + \sqrt{2\epsilon}\;\boldsymbol{z}_k, \qquad \boldsymbol{z}_k \sim \mathcal{N}(\boldsymbol{0}, I).
$$

Intuition, term by term:

* $\epsilon\,\nabla \log p$: a small gradient-ascent step **uphill** on the log-density. Alone, this would just climb to a mode and sit there (a MAP estimate, not a sample).
* $\sqrt{2\epsilon}\,\boldsymbol{z}_k$: a random kick. It stops you from collapsing onto the peak and makes you *wander around the hill in proportion to its width*.

The specific ratio (step $\epsilon$ for the drift, standard deviation $\sqrt{2\epsilon}$ for the noise) is what makes the wandering settle into exactly $p$ as its stationary distribution; the factor 2 is not arbitrary — you will see it fall out of the Fokker–Planck equation in Chapter 5. As $\epsilon \to 0$ and $k \to \infty$, $\boldsymbol{x}_k$ is a sample from $p$.

**So the score is the minimum you need to sample from a distribution.** That is why diffusion models are, at their heart, *score estimators*.

---

## Chapter 2 — Why the naive score fails, and how noise rescues it

### 2.1 The manifold problem

Real data (images, audio, proteins) lives on a thin low-dimensional manifold inside $\mathbb{R}^n$. Off the manifold, $p_0(\boldsymbol{x}) \approx 0$, and $\log p_0 \to -\infty$: the score is undefined or wildly inaccurate there. But a Langevin chain has to *start* somewhere, and it starts off-manifold. Also, if you try to learn the score with a neural network from samples, you only ever see training points *on* the manifold, so the network has no idea what the arrow should be in the vast empty regions. The paper says this in Section 1.3: *"we do not get observations outside of the manifold and hence the vector-field estimation is inaccurate in these regions."*

### 2.2 The fix: blur the distribution

Instead of the score of $p_0$, ask for the score of a **noised** version of the data:

$$
\boldsymbol{X}_t = \boldsymbol{X}_0 + \sigma_t \boldsymbol{Z}, \qquad p_t = \text{density of } \boldsymbol{X}_t .
$$

Convolving $p_0$ with a Gaussian of width $\sigma_t$ smears the thin manifold into a thick, fuzzy cloud that has positive density *everywhere*. Now $\nabla \log p_t$ is well-defined and smooth at every point. Big $\sigma_t$ gives a very smooth, almost-Gaussian landscape (easy to learn, but not the true data); small $\sigma_t$ gives a landscape close to the real one (accurate, but only near the manifold).

The diffusion idea: learn the score at *all* noise levels, $\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$ for $t \in [0,T]$, and during sampling walk from the heavily-blurred landscape toward the sharp one. By the time you are asking about the sharp landscape, you are already standing near the manifold where the estimate is trustworthy — the "warm-start" the paper mentions.

### 2.3 The key identity: score = denoising direction (Tweedie's formula)

This is the single most important computation in the whole story, and it is short. With $\boldsymbol{x}_t = \boldsymbol{x}_0 + \sigma_t \boldsymbol{z}$,

$$
p_t(\boldsymbol{x}_t) = \int p_0(\boldsymbol{x}_0)\,\mathcal{N}(\boldsymbol{x}_t;\, \boldsymbol{x}_0, \sigma_t^2 I)\, d\boldsymbol{x}_0 .
$$

Differentiate under the integral, using the Gaussian score from §1.3 ($\nabla_{\boldsymbol{x}_t}\mathcal{N} = \mathcal{N}\cdot\frac{\boldsymbol{x}_0 - \boldsymbol{x}_t}{\sigma_t^2}$):

$$
\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t) = \int p_0(\boldsymbol{x}_0)\,\mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2 I)\,\frac{\boldsymbol{x}_0 - \boldsymbol{x}_t}{\sigma_t^2}\, d\boldsymbol{x}_0 .
$$

Divide both sides by $p_t(\boldsymbol{x}_t)$. The ratio $p_0(\boldsymbol{x}_0)\mathcal{N}(\cdot)/p_t(\boldsymbol{x}_t)$ is by Bayes exactly the posterior $p(\boldsymbol{x}_0 | \boldsymbol{x}_t)$, so the integral becomes a conditional expectation:

$$
\boxed{\;\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) \;=\; \frac{\mathbb{E}[\boldsymbol{X}_0 \mid \boldsymbol{X}_t = \boldsymbol{x}_t] - \boldsymbol{x}_t}{\sigma_t^2}\;}
$$

Read it slowly. The score at a noisy point $\boldsymbol{x}_t$ is **the arrow from $\boldsymbol{x}_t$ to the best possible denoised guess of the clean image**, divided by the noise variance. Compare with §1.3: the Gaussian score was $(\boldsymbol{\mu} - \boldsymbol{x})/\sigma^2$; here the role of the "mean" is played by the posterior mean $\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t]$, i.e. the MMSE denoiser.

Two consequences that unlock everything downstream:

1. **A score network is a denoiser.** Training a network to predict $\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t]$ (or equivalently the noise $\boldsymbol{z}$, since $\boldsymbol{x}_0 = \boldsymbol{x}_t - \sigma_t\boldsymbol{z}$) is the same as training it to output the score. That is why a DDPM's "$\epsilon$-prediction" network **is** a score model: $\nabla \log p_t(\boldsymbol{x}_t) = -\mathbb{E}[\boldsymbol{Z}|\boldsymbol{x}_t]/\sigma_t$. The *denoising score matching* loss is literally
   $$
   \min_\theta\; \mathbb{E}_{t,\,\boldsymbol{x}_0,\,\boldsymbol{z}} \Big\|\, \boldsymbol{s}_\theta(\boldsymbol{x}_0 + \sigma_t\boldsymbol{z}, t) + \frac{\boldsymbol{z}}{\sigma_t} \Big\|^2,
   $$
   and its minimizer is the true score because the conditional expectation is the least-squares optimum.
2. **In the survey, $\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{X}_t = \boldsymbol{x}_t]$ appears everywhere** (look at Figure 1: DPS, ΠGDM, Moment Matching, DDNM all contain it). It is simply "the score, re-expressed as a denoised image". When you see it, mentally substitute $\boldsymbol{x}_t + \sigma_t^2 \nabla\log p_t(\boldsymbol{x}_t)$.

Now you have all the pieces. On to the paper's equations.

---

## Chapter 3 — Equation 2.1: the forward SDE (the corruption, written in continuous time)

$$
d\boldsymbol{x}_t = \underbrace{\boldsymbol{f}(\boldsymbol{x}_t, t)}_{\text{drift}}\,dt + \underbrace{g(t)}_{\text{diffusion}}\,d\boldsymbol{W}_t, \qquad \boldsymbol{x}_0 \sim p_0 . \tag{2.1}
$$

### 3.1 How to read an SDE

An SDE is just a recipe for a tiny time step. Over a small interval $\Delta t$, the increment of Brownian motion $\boldsymbol{W}_{t+\Delta t} - \boldsymbol{W}_t$ is a Gaussian with mean $\boldsymbol{0}$ and covariance $\Delta t\, I$, i.e. it equals $\sqrt{\Delta t}\,\boldsymbol{z}$ with $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},I)$. So Eq. 2.1 *means*:

$$
\boldsymbol{x}_{t+\Delta t} \approx \boldsymbol{x}_t + \boldsymbol{f}(\boldsymbol{x}_t,t)\,\Delta t + g(t)\,\sqrt{\Delta t}\;\boldsymbol{z}, \qquad \boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0}, I).
$$

* **Drift $\boldsymbol{f}$** — the deterministic push. Where would the point go if there were no randomness? (Often $\boldsymbol{0}$, or a pull toward the origin.)
* **Diffusion $g$** — how much fresh noise gets injected per unit time. Note the $\sqrt{\Delta t}$: noise standard deviation grows like the square root of time, variance grows linearly. That is the signature of Brownian motion.

### 3.2 What $p_t$ is

Start a whole *population* of points, one per data sample, and run them all forward with Eq. 2.1. The histogram of the population at time $t$ is $p_t$. It starts as the data distribution $p_0$ and, by design of $\boldsymbol{f}$ and $g$, ends at time $T$ as something indistinguishable from a simple Gaussian, $p_T \approx \mathcal{N}(\boldsymbol{0}, \sigma_T^2 I)$ or $\mathcal{N}(\boldsymbol{0}, I)$. This is the diffusion you already know; 2.1 is just the continuous-time language for it, which lets one equation cover DDPM, NCSN, and every variant at once (the "SDE variants" paragraph picks $\boldsymbol{f}, g$ to recover each).

---

## Chapter 4 — Equation 2.2: the reverse SDE, and why the score is *forced* to appear

$$
d\boldsymbol{x}_t = \Big(\boldsymbol{f}(\boldsymbol{x}_t, t) - g^2(t)\,\underbrace{\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)}_{\text{score}}\Big)\,dt + g(t)\,d\bar{\boldsymbol{W}}_t, \qquad \boldsymbol{x}_T \sim p_T . \tag{2.2}
$$

(Here time runs backwards from $T$ to $0$, so $dt < 0$, and $d\bar{\boldsymbol{W}}_t$ is a Brownian motion in reversed time.)

### 4.1 Why reversing needs *extra* information

Imagine the forward process as ink spreading in water. Run a video of it backwards and you see ink un-spreading back into a drop. But if you only had the *rule* of the forward process ("each molecule jiggles randomly"), you could not reverse it — a random jiggle backwards is still just a random jiggle; it does not know where the drop was. To un-spread, each molecule needs to know **which direction the concentration is higher**, so it can drift back toward the drop. "Which direction is the concentration higher at time $t$" is *exactly* $\nabla \log p_t$. This is Anderson's 1982 theorem in one sentence: the time-reversal of a diffusion is another diffusion whose drift is corrected by the score of the marginal density.

### 4.2 Read the reverse step concretely

Write Eq. 2.2 as a discrete backward step (with $\Delta t > 0$, going from $t$ to $t - \Delta t$):

$$
\boldsymbol{x}_{t-\Delta t} \approx \boldsymbol{x}_t - \Big(\boldsymbol{f}(\boldsymbol{x}_t,t) - g^2(t)\,\nabla\log p_t(\boldsymbol{x}_t)\Big)\Delta t + g(t)\sqrt{\Delta t}\,\boldsymbol{z}
= \boldsymbol{x}_t - \boldsymbol{f}\,\Delta t + g^2\,\Delta t\;\nabla \log p_t(\boldsymbol{x}_t) + g\sqrt{\Delta t}\,\boldsymbol{z}.
$$

Three moves per step:

1. **Undo the drift** ($-\boldsymbol{f}\Delta t$): whatever deterministic push the forward process applied, apply it in reverse.
2. **Climb the score** ($+g^2\Delta t\,\nabla\log p_t$): move toward higher density at the current noise level. By Tweedie (§2.3) this is $+\frac{g^2 \Delta t}{\sigma_t^2}\big(\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t] - \boldsymbol{x}_t\big)$: **take a small step toward the denoised image.** The size of the step is proportional to $g^2$ — the more noise the forward process was injecting at this moment, the more spreading there is to undo, so the harder you pull back.
3. **Re-inject a bit of noise** ($g\sqrt{\Delta t}\boldsymbol{z}$): keep the process stochastic so that you produce *samples* from $p_0$, not a single deterministic point.

Sanity check with the simplest case, $\boldsymbol{f} = \boldsymbol{0}$, $g = 1$: the step becomes $\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t + \Delta t\,\nabla\log p_t(\boldsymbol{x}_t) + \sqrt{\Delta t}\,\boldsymbol{z}$. Compare with Langevin dynamics from §1.4 with $\epsilon = \Delta t$: it is *almost* the same update (drift $\epsilon\, s$, noise $\sqrt{\epsilon}$ instead of $\sqrt{2\epsilon}$), except the target density is slowly sharpening from $p_t$ toward $p_0$ as you go. Reverse diffusion is essentially **annealed Langevin dynamics**. The exact relationship (where the missing factor of 2 went) is explained in §5.3.

### 4.3 Why the initialization is legal

The reverse SDE must start from $\boldsymbol{x}_T \sim p_T$. We do not know $p_T$ exactly, but for large $T$ and *linear* drift the forward process forgets its starting point and $p_T$ becomes a known Gaussian (proved in Chapters 6 and 7 below for the two standard choices). So we just sample $\boldsymbol{x}_T$ from that Gaussian. The only unknown left in Eq. 2.2 is the score — hence the paper's line: *"the remaining goal becomes to estimate the score function."*

---

## Chapter 5 — Equations 2.3 and 2.4: the probability-flow ODE

$$
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}(\boldsymbol{x}_t, t) - \frac{g^2(t)}{2}\,\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t). \tag{2.3}
$$

### 5.1 The claim

There exists a **deterministic** ODE — no noise term at all — whose solutions, if you start them from a population distributed as $p_T$, are distributed as $p_t$ at every intermediate time and as $p_0$ at the end. Same marginals as the SDE, but each particle follows a smooth, repeatable path. Note the coefficient: $\tfrac{1}{2}g^2$ rather than $g^2$.

### 5.2 Where it comes from: the Fokker–Planck equation (the one derivation worth seeing)

The density $p_t$ of the forward SDE (2.1) evolves by the Fokker–Planck equation:

$$
\frac{\partial p_t}{\partial t} = -\nabla\cdot\big(\boldsymbol{f}\,p_t\big) + \frac{g^2}{2}\,\Delta p_t .
$$

The first term is transport by the drift; the second is heat-equation-style spreading by the noise. Now the trick. Because $\nabla p_t = p_t\,\nabla \log p_t$, the diffusion term can be rewritten as a *transport* term:

$$
\frac{g^2}{2}\Delta p_t = \frac{g^2}{2}\nabla\cdot(\nabla p_t) = \nabla\cdot\Big(\frac{g^2}{2}\,p_t\,\nabla\log p_t\Big) = -\nabla\cdot\Big(\underbrace{-\tfrac{g^2}{2}\nabla\log p_t}_{\text{a velocity}}\;p_t\Big).
$$

Substituting back,

$$
\frac{\partial p_t}{\partial t} = -\nabla\cdot\Big(\big[\boldsymbol{f} - \tfrac{g^2}{2}\nabla\log p_t\big]\,p_t\Big),
$$

which is a **continuity equation** (a pure transport equation with no diffusion term) for a fluid with velocity field $\boldsymbol{v}(\boldsymbol{x},t) = \boldsymbol{f} - \tfrac{g^2}{2}\nabla\log p_t$. A fluid element moving with that velocity, $d\boldsymbol{x}/dt = \boldsymbol{v}$, reproduces the same density evolution. That is Eq. 2.3.

Intuition: *random spreading of a density is indistinguishable, at the level of the density, from a deterministic flow that pushes mass downhill on $\log p$ at speed $\tfrac{g^2}{2}$.* Running that flow backward in time pushes mass uphill toward the data — a deterministic denoising trajectory. This is where the factor $\tfrac12$ comes from: it is the $\tfrac12$ in front of the Laplacian in Fokker–Planck, which in turn comes from the second-order Itô term / the fact that Brownian variance is $\Delta t$, not $2\Delta t$.

### 5.3 SDE = ODE + Langevin corrector (why the SDE has $g^2$ and the ODE has $g^2/2$)

Now the missing factor of 2 from §4.2 resolves itself. Split the reverse-SDE drift:

$$
\boldsymbol{f} - g^2\,\nabla\log p_t \;=\; \underbrace{\Big(\boldsymbol{f} - \tfrac{g^2}{2}\nabla\log p_t\Big)}_{\text{probability-flow ODE}} \;+\; \underbrace{\Big(-\tfrac{g^2}{2}\nabla\log p_t\Big)}_{\text{extra drift}} ,
$$

and recall the SDE also has the noise term $g\,d\bar{\boldsymbol{W}}$. The pair "extra drift $\tfrac{g^2}{2}\nabla\log p_t$ (in reverse time) plus noise of size $g$" is *exactly* a Langevin step (§1.4 with $\epsilon = \tfrac{g^2}{2}dt$: drift $\epsilon\,\nabla\log p$, noise $\sqrt{2\epsilon} = g\sqrt{dt}$), and Langevin dynamics leaves $p_t$ invariant. So:

> **Reverse SDE = (deterministic flow that moves the density from $p_t$ to $p_{t-dt}$) + (a Langevin jiggle that does not change the density but reshuffles particles within it).**

The ODE does only the first part; the SDE adds the second. Both have the same marginals. The jiggle is useful in practice because it corrects accumulated errors in the score estimate — this is the "predictor–corrector" idea from Song et al.

### 5.4 Equation 2.4: Euler discretization

Any ODE solver can integrate 2.3 backward from $T$ to $0$. The simplest is Euler, and the paper writes it as

$$
\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t + \Delta t\Big(\boldsymbol{f}(\boldsymbol{x}_t,t) - \frac{g^2(t)}{2}\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\Big). \tag{2.4}
$$

**A note on the sign, so it does not confuse you.** For an ODE $d\boldsymbol{x}/dt = \boldsymbol{v}$, stepping *backward* by a positive $\Delta t$ gives $\boldsymbol{x}_{t-\Delta t} \approx \boldsymbol{x}_t - \Delta t\,\boldsymbol{v}$. With $\boldsymbol{f} = \boldsymbol{0}$ that would be $\boldsymbol{x}_t + \Delta t\,\tfrac{g^2}{2}\nabla\log p_t$: a step *toward* the data, as it should be. The paper's Eq. 2.4 as printed has the opposite sign, which is either a typo or assumes $\Delta t$ is itself negative (a signed step). Do not let it derail your understanding: the meaning is "take a small deterministic step in the direction $-\boldsymbol{f} + \tfrac{g^2}{2}\nabla\log p_t$, i.e. toward higher density". Deterministic samplers such as DDIM are exactly this with better step schedules and integrators.

---

## Chapter 6 — Equation 2.5: the Variance-Exploding (VE) SDE

Choose $\boldsymbol{f}(\boldsymbol{x}_t, t) = \boldsymbol{0}$ (no drift) and $g(t) = \sqrt{\tfrac{d\sigma_t^2}{dt}}$ for some increasing noise schedule $\sigma_t$ with $\sigma_0 = 0$.

### 6.1 Why the forward process is "just add noise"

With no drift, Eq. 2.1 becomes $d\boldsymbol{x}_t = g(t)\,d\boldsymbol{W}_t$, so $\boldsymbol{x}_t = \boldsymbol{x}_0 + \int_0^t g(s)\,d\boldsymbol{W}_s$. A stochastic integral of a deterministic function against Brownian motion is Gaussian with mean $\boldsymbol{0}$ and variance (Itô isometry — "variances of independent increments add")

$$
\int_0^t g(s)^2\,ds = \int_0^t \frac{d\sigma_s^2}{ds}\,ds = \sigma_t^2 - \sigma_0^2 = \sigma_t^2 .
$$

Therefore

$$
\boldsymbol{X}_t = \boldsymbol{X}_0 + \sigma_t\,\boldsymbol{Z}, \qquad \boldsymbol{Z}\sim\mathcal{N}(\boldsymbol{0}, I_n). \tag{2.5}
$$

This is precisely the "smoothed" distribution of Chapter 2, so Tweedie applies verbatim: $\nabla\log p_t(\boldsymbol{x}_t) = \big(\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t] - \boldsymbol{x}_t\big)/\sigma_t^2$. The reason $g$ is defined as $\sqrt{d\sigma_t^2/dt}$ is exactly so that this holds: $g^2$ is the *rate* at which variance is being added, and integrating a rate gives the total.

### 6.2 The special case $\sigma_t = \sqrt{t}$

Then $\sigma_t^2 = t$, $d\sigma_t^2/dt = 1$, $g \equiv 1$: plain Brownian motion. The reverse SDE step is $\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t + \Delta t\,\nabla\log p_t + \sqrt{\Delta t}\,\boldsymbol{z}$ — the annealed-Langevin form from §4.2. Plugging in Tweedie with $\sigma_t^2 = t$:

$$
\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t + \frac{\Delta t}{t}\big(\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t] - \boldsymbol{x}_t\big) + \sqrt{\Delta t}\,\boldsymbol{z} .
$$

Read it: move a fraction $\Delta t / t$ of the way toward the denoised image, then add a little noise. When $t$ is large the fraction is small (trust the denoiser cautiously, it is looking at very noisy input); as $t \to 0$ the fraction grows (the denoiser is reliable, commit to it).

"Variance exploding" because $\sigma_T^2$ is made huge; then $p_T \approx \mathcal{N}(\boldsymbol{0}, \sigma_T^2 I)$ no matter what $p_0$ was (the data is a negligible perturbation of the noise), which gives the legal initialization required in §4.3. This is the NCSN / Song–Ermon family.

---

## Chapter 7 — Equations 2.6 and 2.7: the Variance-Preserving (VP) SDE and the OU process

Now choose a drift that pulls toward the origin: $\boldsymbol{f}(\boldsymbol{x}_t, t) = -\boldsymbol{x}_t$, with $g = \sqrt{2}$. This is the Ornstein–Uhlenbeck process:

$$
d\boldsymbol{x}_t = -\boldsymbol{x}_t\,dt + \sqrt{2}\,d\boldsymbol{W}_t . \tag{2.6}
$$

### 7.1 Intuition before algebra

Two forces on each particle: a spring pulling it toward $\boldsymbol{0}$ with strength proportional to its distance, and random kicks. The spring shrinks the *signal*; the kicks add *noise*. When the two balance, the particle sits in a stationary Gaussian cloud. The balance point for spring constant 1 and noise $\sqrt{2}$ is exactly $\mathcal{N}(\boldsymbol{0}, I)$ — you can verify it by plugging $p = \mathcal{N}(\boldsymbol{0},I)$, whose score is $-\boldsymbol{x}$, into §1.4: Langevin with $\epsilon\,\nabla\log p = -\epsilon\boldsymbol{x}$ and noise $\sqrt{2\epsilon}$ *is* Eq. 2.6 with $\epsilon = dt$. So the OU process is Langevin dynamics targeting the standard normal — the forward process is literally "sampling from $\mathcal{N}(\boldsymbol{0}, I)$ by Langevin, starting from a data point", and we record the trajectory.

### 7.2 Solving it (integrating-factor trick)

Multiply through by $e^{t}$ and notice $d(e^{t}\boldsymbol{x}_t) = e^{t}d\boldsymbol{x}_t + e^{t}\boldsymbol{x}_t\,dt$ (no Itô correction is needed here because $e^t$ is deterministic). Substituting Eq. 2.6,

$$
d(e^{t}\boldsymbol{x}_t) = e^{t}\big(-\boldsymbol{x}_t\,dt + \sqrt{2}\,d\boldsymbol{W}_t\big) + e^{t}\boldsymbol{x}_t\,dt = \sqrt{2}\,e^{t}\,d\boldsymbol{W}_t .
$$

Integrate from $0$ to $t$ and divide by $e^{t}$:

$$
\boldsymbol{x}_t = e^{-t}\boldsymbol{x}_0 + \sqrt{2}\int_0^t e^{-(t-s)}\,d\boldsymbol{W}_s .
$$

The integral is Gaussian, mean $\boldsymbol{0}$, variance (Itô isometry again)

$$
2\int_0^t e^{-2(t-s)}\,ds = 2\cdot\frac{1 - e^{-2t}}{2} = 1 - e^{-2t} .
$$

Hence

$$
\boldsymbol{X}_t = e^{-t}\boldsymbol{X}_0 + \sqrt{1 - e^{-2t}}\;\boldsymbol{Z}, \qquad \boldsymbol{Z}\sim\mathcal{N}(\boldsymbol{0}, I_n). \tag{2.7}
$$

### 7.3 Reading 2.7

* Signal coefficient $e^{-t}$: the clean image fades exponentially.
* Noise standard deviation $\sqrt{1 - e^{-2t}}$: the noise grows to fill the gap. Note $(e^{-t})^2 + (\sqrt{1-e^{-2t}})^2 = 1$: if the data had unit variance, the total variance stays 1 forever — hence **variance preserving**.
* As $t \to \infty$: $\boldsymbol{X}_t \to \mathcal{N}(\boldsymbol{0}, I)$ regardless of $\boldsymbol{X}_0$. Legal initialization for the reverse process, exactly as in §4.3.
* This is DDPM in disguise: DDPM writes $\boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{z}$; set $\bar\alpha_t = e^{-2t}$ and you have Eq. 2.7. The discrete Markov chain you already know is the Euler discretization of the OU SDE.

### 7.4 The score in the VP case

Tweedie generalizes to any affine corruption $\boldsymbol{x}_t = a_t\boldsymbol{x}_0 + b_t\boldsymbol{z}$ (same derivation as §2.3 with $\mathcal{N}(\boldsymbol{x}_t; a_t\boldsymbol{x}_0, b_t^2 I)$):

$$
\nabla\log p_t(\boldsymbol{x}_t) = \frac{a_t\,\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t] - \boldsymbol{x}_t}{b_t^2} = -\frac{\mathbb{E}[\boldsymbol{Z}|\boldsymbol{x}_t]}{b_t},
\qquad a_t = e^{-t},\; b_t = \sqrt{1-e^{-2t}} .
$$

So the DDPM noise-prediction network $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \approx \mathbb{E}[\boldsymbol{Z}|\boldsymbol{x}_t]$ gives the score as $-\boldsymbol{\epsilon}_\theta/\sqrt{1-\bar\alpha_t}$. Any pretrained DDPM *is* a score model for the VP SDE, and any pretrained NCSN *is* one for the VE SDE. That is why the survey can treat "a pretrained diffusion model" as "an oracle for $\nabla\log p_t$" and never worry about which architecture produced it.

---

## Epilogue — Back to the inverse problem (why the survey needed all this)

You want to reconstruct $\boldsymbol{x}$ from measurements $\boldsymbol{y} = \mathcal{A}(\boldsymbol{x}) + \sigma_{\boldsymbol{y}}\boldsymbol{z}$ by sampling the posterior $p(\boldsymbol{x}|\boldsymbol{y})$. Chapter 1 told you that sampling needs only the score, and Eq. 1.2 told you the posterior score splits as prior score + measurement score. Running the reverse SDE (2.2) with the *conditional* score therefore samples the posterior:

$$
d\boldsymbol{x}_t = \Big(\boldsymbol{f} - g^2\big[\underbrace{\nabla\log p_t(\boldsymbol{x}_t)}_{\text{pretrained diffusion model}} + \underbrace{\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{y}|\boldsymbol{x}_t)}_{\text{the hard part}}\big]\Big)dt + g\,d\bar{\boldsymbol{W}}_t .
$$

The first term is free — it is the model you downloaded. The second is the trouble: at $t = 0$ it is closed-form ($\frac{\boldsymbol{y} - A\boldsymbol{x}}{\sigma_y^2}$ for linear $\mathcal{A}$, from the Gaussian score in §1.3), but at noise level $t$ it requires the intractable integral $p_t(\boldsymbol{y}|\boldsymbol{x}_t) = \int p(\boldsymbol{y}|\boldsymbol{x}_0)\,p(\boldsymbol{x}_0|\boldsymbol{x}_t)\,d\boldsymbol{x}_0$ (Eq. 1.3), because the measurement was taken of the *clean* image but you are standing at a *noisy* one. Every method in Figure 1 is a different approximation of that term, and almost all of them do it by replacing the unknown clean image with the denoised estimate $\mathbb{E}[\boldsymbol{X}_0|\boldsymbol{x}_t]$ — which, by Tweedie, you also get for free from the score. The rest of the survey is a taxonomy of how cleverly each method does that substitution.

---

## Cheat sheet

| Object | Formula | One-line meaning |
|---|---|---|
| Score | $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$ | Compass pointing toward higher probability; ignores normalization |
| Gaussian score | $(\boldsymbol{\mu}-\boldsymbol{x})/\sigma^2$ | Points at the mean, harder when narrower |
| Langevin | $\boldsymbol{x} \leftarrow \boldsymbol{x} + \epsilon\,\boldsymbol{s} + \sqrt{2\epsilon}\,\boldsymbol{z}$ | Climb + jiggle = sample |
| Tweedie | $\nabla\log p_t(\boldsymbol{x}_t) = \dfrac{\mathbb{E}[\boldsymbol{X}_0\|\boldsymbol{x}_t]-\boldsymbol{x}_t}{\sigma_t^2}$ | Score at a noisy point = direction to the denoised point |
| 2.1 forward SDE | $d\boldsymbol{x} = \boldsymbol{f}\,dt + g\,d\boldsymbol{W}$ | Push by $\boldsymbol{f}$, add noise at rate $g^2$ |
| 2.2 reverse SDE | $d\boldsymbol{x} = (\boldsymbol{f} - g^2\boldsymbol{s})\,dt + g\,d\bar{\boldsymbol{W}}$ | Undo push, climb score, re-jiggle |
| 2.3 PF-ODE | $\dot{\boldsymbol{x}} = \boldsymbol{f} - \tfrac{g^2}{2}\boldsymbol{s}$ | Deterministic flow with the same marginals (from Fokker–Planck) |
| 2.4 Euler | $\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t \mp \Delta t(\boldsymbol{f} - \tfrac{g^2}{2}\boldsymbol{s})$ | One solver step (mind the sign convention) |
| 2.5 VE | $\boldsymbol{X}_t = \boldsymbol{X}_0 + \sigma_t\boldsymbol{Z}$ | $\boldsymbol{f}=0$, $g^2 = d\sigma_t^2/dt$; NCSN |
| 2.6 OU | $d\boldsymbol{x} = -\boldsymbol{x}\,dt + \sqrt{2}\,d\boldsymbol{W}$ | Spring to origin + kicks; Langevin toward $\mathcal{N}(0,I)$ |
| 2.7 VP | $\boldsymbol{X}_t = e^{-t}\boldsymbol{X}_0 + \sqrt{1-e^{-2t}}\,\boldsymbol{Z}$ | Signal fades, noise fills in; DDPM with $\bar\alpha_t = e^{-2t}$ |
