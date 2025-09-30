#!/usr/bin/env python3
"""
AI Learning Progress Tracker
Track your progress through the 6-month AI learning program
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

class ProgressTracker:
    def __init__(self, progress_file: str = "progress.json"):
        self.progress_file = progress_file
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load progress from file or create new structure"""
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._initialize_progress()
    
    def _initialize_progress(self) -> Dict:
        """Initialize progress structure"""
        return {
            "start_date": None,
            "current_week": 1,
            "weeks": {
                f"week_{i:02d}": {
                    "month": ((i-1) // 4) + 1,
                    "week_in_month": ((i-1) % 4) + 1,
                    "focus_area": self._get_focus_area(i),
                    "papers": self._get_papers(i),
                    "coding_task": self._get_coding_task(i),
                    "status": "⬜ Not Started",
                    "start_date": None,
                    "completion_date": None,
                    "notes": "",
                    "resources": self._get_resources(i)
                }
                for i in range(1, 25)
            }
        }
    
    def _get_focus_area(self, week: int) -> str:
        """Get focus area for week"""
        focus_areas = [
            "Math for ML", "Neural Nets Basics", "PyTorch Mastery", "Representation Learning",
            "Attention Mechanism", "Transformer Architecture", "Pretraining & Scaling Laws", "Posttraining & Alignment",
            "GANs & VAEs", "Diffusion Models", "Video Generation", "Evaluation",
            "RL Basics", "World Models", "EBMs & JEPA", "RLHF & DPO",
            "Multimodal Models", "Agents & Tools", "Retrieval & Memory", "Systems & Efficiency",
            "AI Safety & Alignment", "Robustness & Interpretability", "Cross-Frontiers", "Capstone"
        ]
        return focus_areas[week - 1]
    
    def _get_papers(self, week: int) -> List[str]:
        """Get papers for week"""
        papers = [
            [],  # Week 1
            ["LeCun et al. (1998)"],  # Week 2
            ["Karpathy 'NN Zero to Hero'"],  # Week 3
            ["SimCLR (2020)"],  # Week 4
            ["Vaswani et al. (2017, Sec. 3–4)"],  # Week 5
            ["Vaswani et al. (2017, full)"],  # Week 6
            ["Kaplan (2020), BERT (2018)"],  # Week 7
            ["InstructGPT (2022), DPO (2023)"],  # Week 8
            ["Goodfellow (2014), Kingma (2013)"],  # Week 9
            ["Ho (2020), Rombach (2022)"],  # Week 10
            ["Imagen Video (2022)"],  # Week 11
            ["Heusel (2017)"],  # Week 12
            ["Sutton & Barto (Ch. 1–3)"],  # Week 13
            ["Ha & Schmidhuber (2018), DreamerV2"],  # Week 14
            ["LeCun (2022), Grathwohl (2019)"],  # Week 15
            ["Christiano (2017), Ouyang (2022)"],  # Week 16
            ["CLIP (2021), Flamingo (2022)"],  # Week 17
            ["ReAct (2022), Toolformer (2023)"],  # Week 18
            ["RAG (2020), Atlas (2022)"],  # Week 19
            ["ZeRO (2020), vLLM (2023)"],  # Week 20
            ["InstructGPT (2022), ConstAI (2022)"],  # Week 21
            ["Goodfellow (2015), LIME (2016)"],  # Week 22
            ["NeRF (2020), AlphaFold (2021)"],  # Week 23
            ["Pick one (World Model / Multimodal / Alignment / NeRF)"]  # Week 24
        ]
        return papers[week - 1]
    
    def _get_coding_task(self, week: int) -> str:
        """Get coding task for week"""
        tasks = [
            "Linear/Logistic Regression from scratch",
            "2-layer MLP on MNIST (NumPy)",
            "CNN on MNIST (PyTorch)",
            "Autoencoder + contrastive CIFAR-10",
            "Implement scaled dot-product attention",
            "Train toy Transformer (translation/text)",
            "Train mini GPT-2 on WikiText-2",
            "Fine-tune GPT-2 with Alpaca (LoRA)",
            "Train DCGAN + VAE (CIFAR-10)",
            "Minimal diffusion on MNIST",
            "Toy video diffusion model",
            "Compare FID of GAN/VAE/Diffusion",
            "Q-learning + Policy Gradient (CartPole)",
            "Train VAE+RNN world model (CarRacing)",
            "Train toy EBM on CIFAR-10",
            "Fine-tune GPT-2 with DPO",
            "Fine-tune CLIP for retrieval",
            "Build agent with calculator + wiki API",
            "Build RAG chatbot with FAISS",
            "Quantize GPT-2, serve with vLLM",
            "Fine-tune LLM for harmless/helpful",
            "FGSM attack on MNIST + saliency maps",
            "Train toy NeRF on small 3D dataset",
            "Final project"
        ]
        return tasks[week - 1]
    
    def _get_resources(self, week: int) -> str:
        """Get resources directory for week"""
        resources = [
            "math_ml", "neural_networks", "pytorch_mastery", "representation_learning",
            "attention", "transformers", "pretraining", "alignment",
            "generative_models", "diffusion", "video_generation", "evaluation",
            "reinforcement_learning", "world_models", "ebm_jepa", "rlhf",
            "multimodal", "agents", "retrieval", "systems",
            "ai_safety", "robustness", "cross_frontiers", "capstone"
        ]
        return resources[week - 1]
    
    def start_week(self, week: int) -> None:
        """Start a new week"""
        if 1 <= week <= 24:
            week_key = f"week_{week:02d}"
            self.progress["weeks"][week_key]["status"] = "🟡 In Progress"
            self.progress["weeks"][week_key]["start_date"] = datetime.now().isoformat()
            self.progress["current_week"] = week
            if not self.progress["start_date"]:
                self.progress["start_date"] = datetime.now().isoformat()
            self.save_progress()
            print(f"Started Week {week}: {self.progress['weeks'][week_key]['focus_area']}")
        else:
            print("Invalid week number. Must be between 1 and 24.")
    
    def complete_week(self, week: int, notes: str = "") -> None:
        """Mark a week as completed"""
        if 1 <= week <= 24:
            week_key = f"week_{week:02d}"
            self.progress["weeks"][week_key]["status"] = "✅ Completed"
            self.progress["weeks"][week_key]["completion_date"] = datetime.now().isoformat()
            if notes:
                self.progress["weeks"][week_key]["notes"] = notes
            self.save_progress()
            print(f"Completed Week {week}: {self.progress['weeks'][week_key]['focus_area']}")
        else:
            print("Invalid week number. Must be between 1 and 24.")
    
    def update_notes(self, week: int, notes: str) -> None:
        """Update notes for a week"""
        if 1 <= week <= 24:
            week_key = f"week_{week:02d}"
            self.progress["weeks"][week_key]["notes"] = notes
            self.save_progress()
            print(f"Updated notes for Week {week}")
        else:
            print("Invalid week number. Must be between 1 and 24.")
    
    def get_week_info(self, week: int) -> Dict:
        """Get information for a specific week"""
        if 1 <= week <= 24:
            week_key = f"week_{week:02d}"
            return self.progress["weeks"][week_key]
        else:
            print("Invalid week number. Must be between 1 and 24.")
            return {}
    
    def get_progress_summary(self) -> Dict:
        """Get overall progress summary"""
        total_weeks = 24
        completed = sum(1 for week in self.progress["weeks"].values() if week["status"] == "✅ Completed")
        in_progress = sum(1 for week in self.progress["weeks"].values() if week["status"] == "🟡 In Progress")
        not_started = total_weeks - completed - in_progress
        
        return {
            "total_weeks": total_weeks,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": not_started,
            "completion_percentage": (completed / total_weeks) * 100,
            "current_week": self.progress["current_week"]
        }
    
    def print_progress_table(self) -> None:
        """Print progress table"""
        print("\n" + "="*100)
        print("AI LEARNING PROGRESS TRACKER")
        print("="*100)
        print(f"{'Week':<4} {'Month':<5} {'Focus Area':<25} {'Status':<15} {'Start Date':<12} {'Completion':<12}")
        print("-"*100)
        
        for i in range(1, 25):
            week_info = self.get_week_info(i)
            month = week_info["month"]
            focus = week_info["focus_area"][:24]  # Truncate for display
            status = week_info["status"]
            start = week_info["start_date"][:10] if week_info["start_date"] else ""
            completion = week_info["completion_date"][:10] if week_info["completion_date"] else ""
            
            print(f"{i:<4} {month:<5} {focus:<25} {status:<15} {start:<12} {completion:<12}")
        
        # Print summary
        summary = self.get_progress_summary()
        print("-"*100)
        print(f"Progress: {summary['completed']}/{summary['total_weeks']} weeks completed ({summary['completion_percentage']:.1f}%)")
        print(f"Current Week: {summary['current_week']}")
        print("="*100)
    
    def save_progress(self) -> None:
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

def main():
    """Main function for command-line usage"""
    tracker = ProgressTracker()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "start" and len(sys.argv) > 2:
            week = int(sys.argv[2])
            tracker.start_week(week)
        elif command == "complete" and len(sys.argv) > 2:
            week = int(sys.argv[2])
            notes = sys.argv[3] if len(sys.argv) > 3 else ""
            tracker.complete_week(week, notes)
        elif command == "notes" and len(sys.argv) > 3:
            week = int(sys.argv[2])
            notes = " ".join(sys.argv[3:])
            tracker.update_notes(week, notes)
        elif command == "week" and len(sys.argv) > 2:
            week = int(sys.argv[2])
            info = tracker.get_week_info(week)
            print(f"\nWeek {week} Information:")
            print(f"Focus Area: {info['focus_area']}")
            print(f"Papers: {', '.join(info['papers'])}")
            print(f"Coding Task: {info['coding_task']}")
            print(f"Status: {info['status']}")
            print(f"Resources: src/{info['resources']}/")
            if info['notes']:
                print(f"Notes: {info['notes']}")
        elif command == "summary":
            summary = tracker.get_progress_summary()
            print(f"\nProgress Summary:")
            print(f"Completed: {summary['completed']}/{summary['total_weeks']} weeks")
            print(f"Completion: {summary['completion_percentage']:.1f}%")
            print(f"Current Week: {summary['current_week']}")
        else:
            tracker.print_progress_table()
    else:
        tracker.print_progress_table()

if __name__ == "__main__":
    main()
