# AI Learning Syllabus - 6 Month Research Program

## 📚 Learning Roadmap

### Month 1: Foundations & Neural Networks
**Weeks 1-4: Mathematical Foundations to Neural Networks**

#### Week 1: Math for ML
- **Focus**: Mathematical foundations for machine learning
- **Papers**: None (focus on implementation)
- **Coding Task**: Linear/Logistic Regression from scratch
- **Status**: ⬜ Not Started
- **Resources**: Create `math_ml/` package

#### Week 2: Neural Networks Basics
- **Focus**: Understanding neural network fundamentals
- **Papers**: LeCun et al. (1998) - "Gradient-based learning applied to document recognition"
- **Coding Task**: 2-layer MLP on MNIST (NumPy)
- **Status**: ⬜ Not Started
- **Resources**: Create `neural_networks/` package

#### Week 3: PyTorch Mastery
- **Focus**: Deep learning framework proficiency
- **Papers**: Karpathy "Neural Networks: Zero to Hero" series
- **Coding Task**: CNN on MNIST (PyTorch)
- **Status**: ⬜ Not Started
- **Resources**: Create `pytorch_mastery/` package

#### Week 4: Representation Learning
- **Focus**: Self-supervised learning and contrastive methods
- **Papers**: SimCLR (2020) - "A Simple Framework for Contrastive Learning"
- **Coding Task**: Autoencoder + contrastive learning on CIFAR-10
- **Status**: ⬜ Not Started
- **Resources**: Create `representation_learning/` package

### Month 2: Transformers & Language Models
**Weeks 5-8: Attention to Large Language Models**

#### Week 5: Attention Mechanism
- **Focus**: Understanding attention mechanisms
- **Papers**: Vaswani et al. (2017, Sec. 3–4) - "Attention Is All You Need"
- **Coding Task**: Implement scaled dot-product attention
- **Status**: ⬜ Not Started
- **Resources**: Create `attention/` package

#### Week 6: Transformer Architecture
- **Focus**: Complete transformer implementation
- **Papers**: Vaswani et al. (2017, full) - "Attention Is All You Need"
- **Coding Task**: Train toy Transformer (translation/text)
- **Status**: ⬜ Not Started
- **Resources**: Create `transformers/` package

#### Week 7: Pretraining & Scaling Laws
- **Focus**: Language model pretraining and scaling
- **Papers**: Kaplan (2020) - "Scaling Laws for Neural Language Models", BERT (2018)
- **Coding Task**: Train mini GPT-2 on WikiText-2
- **Status**: ⬜ Not Started
- **Resources**: Create `pretraining/` package

#### Week 8: Posttraining & Alignment
- **Focus**: Model alignment and instruction following
- **Papers**: InstructGPT (2022), DPO (2023) - "Direct Preference Optimization"
- **Coding Task**: Fine-tune GPT-2 with Alpaca (LoRA)
- **Status**: ⬜ Not Started
- **Resources**: Create `alignment/` package

### Month 3: Generative Models
**Weeks 9-12: GANs, VAEs, and Diffusion Models**

#### Week 9: GANs & VAEs
- **Focus**: Generative adversarial networks and variational autoencoders
- **Papers**: Goodfellow (2014) - "Generative Adversarial Networks", Kingma (2013) - "Auto-Encoding Variational Bayes"
- **Coding Task**: Train DCGAN + VAE on CIFAR-10
- **Status**: ⬜ Not Started
- **Resources**: Create `generative_models/` package

#### Week 10: Diffusion Models
- **Focus**: Denoising diffusion probabilistic models
- **Papers**: Ho (2020) - "Denoising Diffusion Probabilistic Models", Rombach (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- **Coding Task**: Minimal diffusion model on MNIST
- **Status**: ⬜ Not Started
- **Resources**: Create `diffusion/` package

#### Week 11: Video Generation
- **Focus**: Video generation and temporal modeling
- **Papers**: Imagen Video (2022) - "Imagen Video: High Definition Video Generation with Diffusion Models"
- **Coding Task**: Toy video diffusion model
- **Status**: ⬜ Not Started
- **Resources**: Create `video_generation/` package

#### Week 12: Evaluation
- **Focus**: Generative model evaluation metrics
- **Papers**: Heusel (2017) - "GANs Trained by a Two Time-Scale Update Rule"
- **Coding Task**: Compare FID of GAN/VAE/Diffusion models
- **Status**: ⬜ Not Started
- **Resources**: Create `evaluation/` package

### Month 4: Reinforcement Learning & World Models
**Weeks 13-16: RL, World Models, and Energy-Based Models**

#### Week 13: RL Basics
- **Focus**: Reinforcement learning fundamentals
- **Papers**: Sutton & Barto (Ch. 1–3) - "Reinforcement Learning: An Introduction"
- **Coding Task**: Q-learning + Policy Gradient on CartPole
- **Status**: ⬜ Not Started
- **Resources**: Create `reinforcement_learning/` package

#### Week 14: World Models
- **Focus**: World models and model-based RL
- **Papers**: Ha & Schmidhuber (2018) - "World Models", DreamerV2
- **Coding Task**: Train VAE+RNN world model on CarRacing
- **Status**: ⬜ Not Started
- **Resources**: Create `world_models/` package

#### Week 15: EBMs & JEPA
- **Focus**: Energy-based models and joint embedding predictive architecture
- **Papers**: LeCun (2022) - "A Path Towards Autonomous Machine Intelligence", Grathwohl (2019) - "Your Classifier is Secretly an Energy Based Model"
- **Coding Task**: Train toy EBM on CIFAR-10
- **Status**: ⬜ Not Started
- **Resources**: Create `ebm_jepa/` package

#### Week 16: RLHF & DPO
- **Focus**: Reinforcement learning from human feedback
- **Papers**: Christiano (2017) - "Deep Reinforcement Learning from Human Preferences", Ouyang (2022) - "Training language models to follow instructions"
- **Coding Task**: Fine-tune GPT-2 with DPO
- **Status**: ⬜ Not Started
- **Resources**: Create `rlhf/` package

### Month 5: Multimodal & Advanced Systems
**Weeks 17-20: Multimodal Models, Agents, and Systems**

#### Week 17: Multimodal Models
- **Focus**: Vision-language models and cross-modal learning
- **Papers**: CLIP (2021) - "Learning Transferable Visual Representations", Flamingo (2022) - "Flamingo: a Visual Language Model for Few-Shot Learning"
- **Coding Task**: Fine-tune CLIP for retrieval
- **Status**: ⬜ Not Started
- **Resources**: Create `multimodal/` package

#### Week 18: Agents & Tools
- **Focus**: AI agents and tool usage
- **Papers**: ReAct (2022) - "ReAct: Synergizing Reasoning and Acting", Toolformer (2023) - "Toolformer: Language Models Can Teach Themselves to Use Tools"
- **Coding Task**: Build agent with calculator + wiki API
- **Status**: ⬜ Not Started
- **Resources**: Create `agents/` package

#### Week 19: Retrieval & Memory
- **Focus**: Retrieval-augmented generation and memory systems
- **Papers**: RAG (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", Atlas (2022) - "Atlas: Few-shot Learning with Retrieval Augmented Language Models"
- **Coding Task**: Build RAG chatbot with FAISS
- **Status**: ⬜ Not Started
- **Resources**: Create `retrieval/` package

#### Week 20: Systems & Efficiency
- **Focus**: Model efficiency and serving systems
- **Papers**: ZeRO (2020) - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", vLLM (2023) - "Efficient Memory Management for Large Language Model Serving"
- **Coding Task**: Quantize GPT-2, serve with vLLM
- **Status**: ⬜ Not Started
- **Resources**: Create `systems/` package

### Month 6: Safety, Robustness & Capstone
**Weeks 21-24: AI Safety, Robustness, and Final Project**

#### Week 21: AI Safety & Alignment
- **Focus**: AI safety and alignment research
- **Papers**: InstructGPT (2022), Constitutional AI (2022) - "Constitutional AI: Harmlessness from AI Feedback"
- **Coding Task**: Fine-tune LLM for harmless/helpful behavior
- **Status**: ⬜ Not Started
- **Resources**: Create `ai_safety/` package

#### Week 22: Robustness & Interpretability
- **Focus**: Model robustness and interpretability
- **Papers**: Goodfellow (2015) - "Explaining and Harnessing Adversarial Examples", LIME (2016) - "Why Should I Trust You?"
- **Coding Task**: FGSM attack on MNIST + saliency maps
- **Status**: ⬜ Not Started
- **Resources**: Create `robustness/` package

#### Week 23: Cross-Frontiers
- **Focus**: Emerging AI frontiers
- **Papers**: NeRF (2020) - "NeRF: Representing Scenes as Neural Radiance Fields", AlphaFold (2021) - "Highly accurate protein structure prediction"
- **Coding Task**: Train toy NeRF on small 3D dataset
- **Status**: ⬜ Not Started
- **Resources**: Create `cross_frontiers/` package

#### Week 24: Capstone
- **Focus**: Final project integration
- **Papers**: Pick one focus area (World Model / Multimodal / Alignment / NeRF)
- **Coding Task**: Final project
- **Status**: ⬜ Not Started
- **Resources**: Create `capstone/` package

## 📊 Progress Tracking

| Month | Week | Focus Area | Papers to Read | Coding Task | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | Math for ML | None | Linear/Logistic Regression from scratch | ⬜ Not Started |  |
| 1 | 2 | Neural Nets Basics | LeCun et al. (1998) | 2-layer MLP on MNIST (NumPy) | ⬜ Not Started |  |
| 1 | 3 | PyTorch Mastery | Karpathy "NN Zero to Hero" | CNN on MNIST (PyTorch) | ⬜ Not Started |  |
| 1 | 4 | Representation Learning | SimCLR (2020) | Autoencoder + contrastive CIFAR-10 | ⬜ Not Started |  |
| 2 | 1 | Attention Mechanism | Vaswani et al. (2017, Sec. 3–4) | Implement scaled dot-product attention | ⬜ Not Started |  |
| 2 | 2 | Transformer Architecture | Vaswani et al. (2017, full) | Train toy Transformer (translation/text) | ⬜ Not Started |  |
| 2 | 3 | Pretraining & Scaling Laws | Kaplan (2020), BERT (2018) | Train mini GPT-2 on WikiText-2 | ⬜ Not Started |  |
| 2 | 4 | Posttraining & Alignment | InstructGPT (2022), DPO (2023) | Fine-tune GPT-2 with Alpaca (LoRA) | ⬜ Not Started |  |
| 3 | 1 | GANs & VAEs | Goodfellow (2014), Kingma (2013) | Train DCGAN + VAE (CIFAR-10) | ⬜ Not Started |  |
| 3 | 2 | Diffusion Models | Ho (2020), Rombach (2022) | Minimal diffusion on MNIST | ⬜ Not Started |  |
| 3 | 3 | Video Generation | Imagen Video (2022) | Toy video diffusion model | ⬜ Not Started |  |
| 3 | 4 | Evaluation | Heusel (2017) | Compare FID of GAN/VAE/Diffusion | ⬜ Not Started |  |
| 4 | 1 | RL Basics | Sutton & Barto (Ch. 1–3) | Q-learning + Policy Gradient (CartPole) | ⬜ Not Started |  |
| 4 | 2 | World Models | Ha & Schmidhuber (2018), DreamerV2 | Train VAE+RNN world model (CarRacing) | ⬜ Not Started |  |
| 4 | 3 | EBMs & JEPA | LeCun (2022), Grathwohl (2019) | Train toy EBM on CIFAR-10 | ⬜ Not Started |  |
| 4 | 4 | RLHF & DPO | Christiano (2017), Ouyang (2022) | Fine-tune GPT-2 with DPO | ⬜ Not Started |  |
| 5 | 1 | Multimodal Models | CLIP (2021), Flamingo (2022) | Fine-tune CLIP for retrieval | ⬜ Not Started |  |
| 5 | 2 | Agents & Tools | ReAct (2022), Toolformer (2023) | Build agent with calculator + wiki API | ⬜ Not Started |  |
| 5 | 3 | Retrieval & Memory | RAG (2020), Atlas (2022) | Build RAG chatbot with FAISS | ⬜ Not Started |  |
| 5 | 4 | Systems & Efficiency | ZeRO (2020), vLLM (2023) | Quantize GPT-2, serve with vLLM | ⬜ Not Started |  |
| 6 | 1 | AI Safety & Alignment | InstructGPT (2022), ConstAI (2022) | Fine-tune LLM for harmless/helpful | ⬜ Not Started |  |
| 6 | 2 | Robustness & Interpretability | Goodfellow (2015), LIME (2016) | FGSM attack on MNIST + saliency maps | ⬜ Not Started |  |
| 6 | 3 | Cross-Frontiers | NeRF (2020), AlphaFold (2021) | Train toy NeRF on small 3D dataset | ⬜ Not Started |  |
| 6 | 4 | Capstone | Pick one (World Model / Multimodal / Alignment / NeRF) | Final project | ⬜ Not Started |  |
