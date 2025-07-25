# DOMA-SMOL

Below is the model composition of each set
## SET 1

| Model                              | Domain        |
|------------------------------------|---------------|
| google/gemma-2-2b-it               | Medical       |
| google/gemma-2-2b-it               | Fitness       |
| google/gemma-2-2b-it               | Mental Health |
| meta-llama/Llama-3.2-3B-Instruct   | Medical       |
| meta-llama/Llama-3.2-3B-Instruct   | Fitness       |
| meta-llama/Llama-3.2-3B-Instruct   | Mental Health |


## SET 2

| Model                              | Domain        |
|------------------------------------|---------------|
| google/gemma-2-2b-it               | Medical       |
| Qwen2.5-3B-Instruct                | Fitness       |
| meta-llama/Llama-3.2-1B-Instruct   | Mental Health |
| microsoft/Phi-3-mini-4k-instruct   | Medical       |
| meta-llama/Llama-3.2-1B-Instruct   | Fitness       |
| meta-llama/Llama-3.2-3B-Instruct   | Mental Health |

## SET 3

| Model                              | Domain        |
|------------------------------------|---------------|
| google/gemma-2-2b-it               | Medical       |
| google/gemma-2-2b-it               | Legal         |
| meta-llama/Llama-3.2-1B-Instruct   | Finance       |
| microsoft/Phi-3-mini-4k-instruct   | Medical       |
| microsoft/Phi-3-mini-4k-instruct   | Finance       |
| microsoft/Phi-3-mini-4k-instruct   | Legal         |


## Sample Queries:

### Medical

```Can you explain how CRISPR technology works and what its implications are for gene editing? I'm especially interested in how it makes changes to DNA and what this could mean for medicine, agriculture, and research. Also, are there any ethical concerns I should be aware of?```

Dataset link - https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT


### Fitness

```I’ve been reading The Complete Guide to Stretching recently in an effort to help find some solutions to muscle aching problems I have. The hypothesis I’ve been operating under is that I have muscle imbalances. And I’ve been generally trying to improve my fitness for a while now. The book says that to correct muscle imbalances you should find and stretch the muscles with poor flexibility but adequate strength from continued mid-range use. And find and strengthen the lax muscles with good flexibility and low strength from continued outer-range use by exercising them within the mid-range. So I want to do is devise a program for myself where I do that. The book has a number of flexibility tests that I’ve done and graded—and yes, I do indeed have some muscles with poor flexibility and some with very good flexibility—but I’m struggling to interpret exactly what the results of these tests mean, in terms of what exercises I should be doing and for how long. I think what would be useful is to have the tests unified with the exercises in some way. In other words: I would just like to have a bunch of flexibility exercises to do where I’m told “when you can do X” you have achieved a good-level of flexibility and you don’t need to push beyond this. So just doing the exercise also acts as the test, telling me when I no longer need to continue. (Even better would be if I’m told that based on a few personal criteria like my height or the length of my limbs etc. which I think you need to take into account to give that information accurately.) Is there anywhere you can get that information? It seems like quite a basic thing to want, to me (just to know where one should stop when doing stretching exercises for optimal normal overall body flexibility). Edit: When I ask how far I should go I don’t mean "how much force should I be using". I know not to go beyond what I’m capable of. So what I‘m asking is: when is what I’m already capable of sufficient.```

Dataset link - https://huggingface.co/datasets/mlfoundations-dev/stackexchange_fitness

### Mental health

```I feel like I'm trying to convince myself that I'm okay when I'm not. I'm always blocking out the bad things and forgetting. I also feel like nobody cares for me and they never will. I feel truly alone.```

Dataset link - https://huggingface.co/datasets/marmikpandya/mental-health

### Legal

```How do I write an accident report if no police officer is available and I need to document the car accident myself? What details should I include to ensure it’s complete for insurance and legal purposes?```

Dataset link - https://huggingface.co/datasets/dzunggg/legal-qa-v1

### Finance

```What kinds of scams or tricks do dealers use when offering 0% financing vs a cash rebate on a car? How can I tell which option actually saves me more money, and what should I watch out for during negotiations?```

Dataset link - https://huggingface.co/datasets/Marina-C/question-answer-Subject-Finance-Instruct/viewer/default/train#:~:text=Marina%2DC/question%2Danswer,Instruct%20%C2%B7%20Datasets%20at%20Hugging%20Face
