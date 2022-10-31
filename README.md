# dpa-vs-clean-label
BAIR assignment 11/3

Both folders contain a dedicated README for the relevant model generation architecture.

We use the clean-label attack method (https://arxiv.org/abs/1912.02771) to generate a poisoned dataset on the MNIST dataset. Testing with poisoning_trigger = all-corners enabled. When generating the poisoned samples, we use the adversarial-sample based method (not the GAN one) and use the autoattack to generate the adversarial perturbation. And then train a model on this poisoned set and see the attack success rate and clean accuracy of the trained model. 

Finally, use DPA (https://arxiv.org/pdf/2006.14768.pdf) to train a robust model on the poisoned samples, and report the attack success rate and clean accuracy.
