# Section 6 - Interpretation and Headline Results

1. Headline Parkinson HRMSE winners:
   - h=1: GARCHND (art_growth, kappa=30%) has the lowest HRMSE (5.3900).
   - h=5: GARCH(1,1) has the lowest HRMSE (2.9825).
   - h=22: GARCHAND (tone) has the lowest HRMSE (2.6020).

2. Significance versus the GARCH(1,1) baseline:
   - DM h=1: significant augmented improvements are GARCH-X (tone) (SE, p=0.000), GARCHAND (tone) (SE, p=0.000), GARCHAND (art_growth) (SE, p=0.001), GARCHND (tone, kappa=30%) (SE, p=0.000), EGARCH-X (tone + art_growth) (SE, p=0.000), GARCH-X (tone) (HRMSE, p=0.000), GARCHAND (tone) (HRMSE, p=0.000), GARCHAND (art_growth) (HRMSE, p=0.027), EGARCH-X (tone + art_growth) (HRMSE, p=0.000).
   - CW h=1: significant augmented improvements are GARCHND (art_growth, kappa=30%) (p=0.018).
   - DM h=5: significant augmented improvements are GARCH-X (tone) (SE, p=0.000), GARCH-X (art_growth) (SE, p=0.000), GARCHAND (tone) (SE, p=0.002), GARCHAND (art_growth) (SE, p=0.000), GARCHND (tone, kappa=30%) (SE, p=0.000), GARCHND (art_growth, kappa=30%) (SE, p=0.011), GARCH-X (tone) (HRMSE, p=0.000), GARCH-X (art_growth) (HRMSE, p=0.000), GARCHAND (tone) (HRMSE, p=0.021), GARCHAND (art_growth) (HRMSE, p=0.000), GARCHND (art_growth, kappa=30%) (HRMSE, p=0.000), EGARCH-X (tone + art_growth) (HRMSE, p=0.001).
   - CW h=5: no augmented model significantly improves adjusted MSPE at 5%.
   - DM h=22: significant augmented improvements are GARCH-X (tone) (SE, p=0.000), GARCH-X (art_growth) (SE, p=0.000), GARCHAND (art_growth) (SE, p=0.000), GARCHND (tone, kappa=30%) (SE, p=0.029), GARCHND (art_growth, kappa=30%) (SE, p=0.000), GARCH-X (tone) (HRMSE, p=0.000), GARCH-X (art_growth) (HRMSE, p=0.000), GARCHAND (art_growth) (HRMSE, p=0.000), GARCHND (art_growth, kappa=30%) (HRMSE, p=0.000).
   - CW h=22: no augmented model significantly improves adjusted MSPE at 5%.

3. MCS survivors at the 90% level against Parkinson HRMSE loss:
   - h=1: GARCH(1,1), GARCH-X (art_growth), GARCHAND (art_growth), GARCHND (tone, kappa=30%), GARCHND (art_growth, kappa=30%)
   - h=5: GARCH(1,1), GARCHND (tone, kappa=30%)
   - h=22: GARCH(1,1), GARCHAND (tone), GARCHND (tone, kappa=30%), EGARCH-X (tone + art_growth)

4. Slow-moving-sentiment hypothesis:
   - Best augmented HRMSE improvement over baseline is h=1: 0.26%, h=5: -0.64%, h=22: 2.50%. The evidence is mixed rather than a clean longer-horizon pattern.

5. Lead-lag diagnostic consistency:
   - The diagnostic says tone_mean_x100_winsor peaks at k=-3; abs_tone_mean_x100_winsor peaks at k=+10; art_growth_winsor peaks at k=-2. Since the OOS evidence is not a clean longer-horizon improvement, the diagnostic should be read as context rather than confirmation.

Total runtime for this run: 81.10 minutes.