# Demo Script

## 90-Second Flow

1. "Real pitchers are not deterministic. They throw distributions."
2. Pick a pitcher, for example Gerrit Cole.
3. Show his real fastball cloud: velocity, release point, movement, and plate location.
4. Toggle `Humanized Mode`.
5. Show 15 sampled trajectories beside 15 canonical repeated trajectories.
6. Show validation: "The held-out classifier AUC is near 0.5, meaning the simulated samples are hard to distinguish from real held-out pitches."
7. Toggle a count bucket: "The distribution shifts by situation."
8. Export the machine-session JSON.
9. Hand over the one-page PDF.

## Practice Sentence

> target system can replicate pitch trajectories. I built a probabilistic layer that learns a pitcher's natural pitch-to-pitch variability from Statcast, samples realistic sessions from that distribution, and validates whether the simulated pitches are statistically hard to distinguish from real held-out pitches.

## If Asked Whether target system Already Does This

Say:

> They may already support playlists or random selection from real pitch sets. What I built is different: an explicit ML layer that quantifies the variability envelope, samples from the learned joint distribution, validates realism, and identifies which dimensions make a session feel fake.

## If Asked Why GMM

Say:

> GMMs are a good first model because a pitch type is not one clean blob. A fastball can have multiple modes: normal competitive fastball, elevated chase fastball, and missed arm-side fastball. A GMM preserves those sub-clouds and the correlations between release, spin, movement, and command.

## If Asked Why Monte Carlo

Say:

> The GMM learns the shape of the pitcher's distribution. Monte Carlo sampling turns that learned shape into actual pitch targets target system could throw.

