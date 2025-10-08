import os
import pandas as pd
import numpy as np
from collections import Counter
import re
import time
import argparse
from typing import List, Dict, Any, Tuple
from google import genai
from tqdm import tqdm
from google.genai import types


GEMINI_API_COSTS_03122025 = {
    "gemini-2.0-flash": {
        "input_price_per_million_tokens": 0.10,
        "output_price_per_million_tokens": 0.40,
        "context_caching_price_per_million_tokens": 0.025,
        "availability_date": "2025-03-31",
    },
    "gemini-2.0-flash-lite": {
        "input_price_per_million_tokens": 0.019,
        "output_price_per_million_tokens": 0.076,
        "availability_date": "2025-03-31",
    },
    "gemini-1.5-flash": {
        "input_price_per_million_tokens": 0.075,
        "output_price_per_million_tokens": 0.30,
        "context_caching_price_per_million_tokens": 0.01875,
    },
    "gemini-1.5-flash-8b": {
        "input_price_per_million_tokens": 0.0375,
        "output_price_per_million_tokens": 0.15,
        "context_caching_price_per_million_tokens": 0.01,
    },
    "gemini-1.5-pro": {
        "input_price_per_million_tokens": 1.25,
        "output_price_per_million_tokens": 5.00,
        "context_caching_price_per_million_tokens": 0.3125,
    },
}

SYSTEM_INSTRUCTION = "You are a coding expert. When I ask you to output data with JSON formatting, please return only correctly formatted JSON without an explanation."


class InstructionAnalyzer:
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.0-flash-lite"):
        """
        Initialize the InstructionAnalyzer with Gemini API credentials.

        Args:
            api_key: Gemini API key
            model_name: Gemini model to use (default: models/gemini-1.5-pro)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

        # Cost estimation parameters (prices as of March 2025)
        # Note: These prices should be verified before use
        self.input_price_per_1k = (
            GEMINI_API_COSTS_03122025[self.model_name.split("/")[1]][
                "input_price_per_million_tokens"
            ]
            / 1000
        )
        self.output_price_per_1k = (
            GEMINI_API_COSTS_03122025[self.model_name.split("/")[1]][
                "output_price_per_million_tokens"
            ]
            / 1000
        )

        # Token estimator: rough approximation (4 chars ≈ 1 token)
        self.chars_per_token = 4

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        return len(text) // self.chars_per_token + 1

    def estimate_cost(
        self, instructions: List[str], expected_output_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Estimate the cost of analyzing the instructions with Gemini.

        Args:
            instructions: List of instruction text strings
            expected_output_tokens: Expected number of tokens in each output response

        Returns:
            Dictionary with cost estimation details
        """
        # Basic statistics
        num_instructions = len(instructions)

        # Estimate total input tokens
        total_chars = sum(len(instruction) for instruction in instructions)
        estimated_input_tokens = total_chars // self.chars_per_token

        # Estimate total output tokens
        estimated_output_tokens = num_instructions * expected_output_tokens

        # Calculate estimated costs
        input_cost = (estimated_input_tokens / 1000) * self.input_price_per_1k
        output_cost = (estimated_output_tokens / 1000) * self.output_price_per_1k
        total_cost = input_cost + output_cost

        return {
            "num_instructions": num_instructions,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_estimated_cost_usd": total_cost,
        }

    def analyze_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Analyze a single instruction using Gemini.

        Args:
            instruction: The instruction text string

        Returns:
            Analysis results from Gemini
        """
        prompt = f"""
        Analyze the following Python coding instruction and identify:
        1. Main topics/concepts
        2. Programming libraries/frameworks mentioned or implied
        3. Tools or applications involved
        4. Level of complexity (beginner, intermediate, advanced)
        5. Type of task (data analysis, web development, automation, etc.)
        
        Return your analysis as a JSON object with the following fields:
        - main_topics
        - libraries_frameworks
        - tools_applications
        - all_complexity
        - all_task_types

        Instruction to analyze: "{instruction}"
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.1,
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )

            # Try to extract JSON from the response
            result_text = response.text
            # Find JSON-like content (anything between curly braces)
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)

            # Convert string to dictionary - this is a simplified approach
            # In production, you'd want more robust JSON parsing
            try:
                import json

                result = json.loads(result_text)
            except:
                # Fallback if JSON parsing fails
                result = {
                    "main_topics": extract_list_items(result_text, "topics"),
                    "libraries_frameworks": extract_list_items(
                        result_text, "libraries"
                    ),
                    "tools_applications": extract_list_items(result_text, "tools"),
                    "complexity": extract_simple_field(result_text, "complexity"),
                    "task_type": extract_simple_field(result_text, "task"),
                }

            return result

        except Exception as e:
            print(f"Error analyzing instruction: {e}")
            return {"error": str(e), "instruction": instruction}

    def batch_analyze(
        self, instructions: List[str], batch_size: int = 10, delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of instructions with rate limiting.

        Args:
            instructions: List of instruction text strings
            batch_size: Number of instructions to process before pausing
            delay: Delay in seconds between batches to avoid rate limits

        Returns:
            List of analysis results
        """
        results = []

        for i, instruction in enumerate(tqdm(instructions)):
            # Add delay between batches to avoid rate limiting
            if i > 0 and i % batch_size == 0:
                time.sleep(delay)

            result = self.analyze_instruction(instruction)
            results.append(result)

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate analysis results to identify trends.

        Args:
            results: List of analysis results from batch_analyze

        Returns:
            Dictionary with aggregated insights
        """
        # Initialize counters
        all_topics = []
        all_libraries = []
        all_tools = []
        all_complexity = []
        all_task_types = []

        # Extract data from results
        for result in results:
            if "error" in result:
                continue

            # Extract and flatten lists
            if "main_topics" in result:
                all_topics.extend(to_list(result["main_topics"]))
            if "libraries_frameworks" in result:
                all_libraries.extend(to_list(result["libraries_frameworks"]))
            if "tools_applications" in result:
                all_tools.extend(to_list(result["tools_applications"]))
            if "complexity" in result:
                all_complexity.append(result["complexity"])
            if "task_type" in result:
                all_task_types.append(result["task_type"])

        all_topics = [topic.lower() for topic in all_topics]
        all_libraries = [topic.lower() for topic in all_libraries]
        all_tools = [topic.lower() for topic in all_tools]
        all_complexity = [topic.lower() for topic in all_complexity]
        all_task_types = [topic.lower() for topic in all_task_types]

        # Count occurrences
        topic_counts = Counter(all_topics)
        library_counts = Counter(all_libraries)
        tool_counts = Counter(all_tools)
        complexity_counts = Counter(all_complexity)
        task_type_counts = Counter(all_task_types)

        # Create topic-library relationships for bubble chart
        topic_library_relationships = []
        for result in results:
            if "error" in result:
                continue

            topics = to_list(result.get("main_topics", []))
            libraries = to_list(result.get("libraries_frameworks", []))
            complexity = result.get("all_complexity", "Unknown")

            for topic in topics:
                for library in libraries:
                    topic_library_relationships.append(
                        {"topic": topic, "library": library, "complexity": complexity}
                    )

        return {
            "top_topics": dict(topic_counts.most_common(1000)),
            "top_libraries": dict(library_counts.most_common(1000)),
            "top_tools": dict(tool_counts.most_common(1000)),
            "complexity_distribution": dict(complexity_counts),
            "task_type_distribution": dict(task_type_counts),
            "topic_library_relationships": topic_library_relationships,
        }

    def save_results(
        self,
        results: List[Dict[str, Any]],
        aggregated_results: Dict[str, Any],
        output_file: str = "analysis_results.json",
    ):
        """Save analysis results to a file."""
        import json

        output = {
            "individual_results": results,
            "aggregated_results": aggregated_results,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_file}")


# Helper functions
def to_list(item):
    """Convert item to a list if it's not already."""
    if isinstance(item, list):
        return item
    return [item]


def extract_list_items(text, keyword):
    """Extract list items from text based on keyword."""
    pattern = rf"{keyword}.*?:(.*?)(?:\n\d+\.|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        items = re.findall(r"[-*]?\s*([^,\n]+)", match.group(1))
        return [item.strip() for item in items if item.strip()]
    return []


def extract_simple_field(text, keyword):
    """Extract a simple field value from text."""
    pattern = rf"{keyword}.*?:\s*([^,\n]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def load_instructions(file_path):
    """Load instructions from a file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        # Assume the first column contains instructions
        return df.iloc[:, 0].tolist()
    elif ext == ".json":
        import json

        with open(file_path, "r") as f:
            data = json.load(f)
        # import pdb
        # pdb.set_trace()
        # Handle different JSON structures
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            elif isinstance(data[0], dict) and "instruction" in data[0]:
                return [item["instruction"] for item in data]
        return []
    elif ext == ".txt":
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python code instructions with Gemini"
    )
    parser.add_argument("input_file", help="Path to file containing instructions")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of instructions to process before pausing",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay in seconds between batches"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Sample size (0 for all instructions)"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate cost without running analysis",
    )

    args = parser.parse_args()

    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key must be provided via --api-key or GEMINI_API_KEY environment variable"
        )

    # Load instructions
    instructions = load_instructions(args.input_file)
    print(f"Loaded {len(instructions)} instructions from {args.input_file}")

    # Sample if requested
    if args.sample > 0 and args.sample < len(instructions):
        instructions = np.random.choice(
            instructions, size=args.sample, replace=False
        ).tolist()
        print(f"Sampled {args.sample} instructions for analysis")

    # Initialize analyzer
    analyzer = InstructionAnalyzer(api_key=api_key)

    # Estimate cost
    cost_estimate = analyzer.estimate_cost(instructions)
    print("\nCost Estimation:")
    print(f"Number of instructions: {cost_estimate['num_instructions']}")
    print(f"Estimated input tokens: {cost_estimate['estimated_input_tokens']:,}")
    print(f"Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
    print(f"Estimated input cost: ${cost_estimate['input_cost_usd']:.2f}")
    print(f"Estimated output cost: ${cost_estimate['output_cost_usd']:.2f}")
    print(f"Total estimated cost: ${cost_estimate['total_estimated_cost_usd']:.2f}")

    if args.estimate_only:
        print("\nEstimate-only mode. Exiting without running analysis.")
        return

    # Confirm before proceeding
    if cost_estimate["total_estimated_cost_usd"] > 5.0:
        proceed = input(
            f"\nWarning: Estimated cost is ${cost_estimate['total_estimated_cost_usd']:.2f}. Proceed? (y/n): "
        )
        if proceed.lower() != "y":
            print("Aborted by user.")
            return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis
    print(f"\nAnalyzing {len(instructions)} instructions...")
    results = analyzer.batch_analyze(
        instructions, batch_size=args.batch_size, delay=args.delay
    )
    # Aggregate results
    print("\nAggregating results...")
    aggregated_results = analyzer.aggregate_results(results)

    # Save results
    analyzer.save_results(
        results,
        aggregated_results,
        output_file=f"{args.output_dir}/analysis_results_k{args.sample}.json",
    )


if __name__ == "__main__":
    main()
