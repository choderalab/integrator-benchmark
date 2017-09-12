mkdir quartic_results
echo Collect quartic equilibrium samples:
bsub -J "quartic_eq_sample" < quartic_eq_sample.lsf
echo Collect steady-state histograms:
bsub -J "histograms[1-40]" < quartic_kl_histograms.lsf
echo Compute nonequilibrium free energies:
bsub -w "done(quartic_eq_sample)" -J "validation[1-80]" < quartic_kl_validation.lsf
echo Plot:
bsub -w "done(validation[1-80]) && done(histograms[1-40])" < quartic_kl_plot.lsf