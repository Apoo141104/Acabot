# Problem-solving prompts and hints inspired by George Pólya
problem_solving_prompts = {
    "understand": {
        "prompt": [
            "First, let's make sure we fully understand the problem. What information are we given?",
            "Understanding the problem thoroughly is key. Can you articulate what the problem is asking?",
            "Before we proceed, let's ensure we have a clear understanding of the problem statement."
        ],
        "hints": "To better understand the problem, try identifying known information and any relationships provided."
    },
    "plan": {
        "prompt": [
            "Now, every problem requires a plan of attack. What strategy or approach do you think might work here?",
            "Let's devise a plan together. What steps could we take to tackle this problem?",
            "Brainstorming different approaches can lead us to the most effective plan."
        ],
        "hints": "Consider breaking down the problem into smaller, more manageable steps. What simpler subproblems do you see?"
    },
    "explore": {
        "prompt": [
            "In mathematics, exploration and creativity are crucial. Let's explore different ideas and approaches.",
            "Don't be afraid to explore unconventional methods. Sometimes, they lead to elegant solutions.",
            "Exploring multiple perspectives can often reveal new insights."
        ],
        "hints": "Try experimenting with different techniques or methods to solve the problem."
    },
    "reflect": {
        "prompt": [
            "Finally, reflecting on our progress is important. What have we tried so far, and what have we learned?",
            "Taking a moment to reflect can help us identify our strengths and areas for improvement.",
            "Let's review our progress and consider any adjustments we might need to make."
        ],
        "hints": "Think about what strategies have been successful and what could be improved for future problem-solving."
    },
    "general": {
        "understand": {
            "prompt": [
                "Can you explain the problem in your own words? What are you trying to find or prove?",
                "What specific information is provided in the problem statement? Are there any conditions or constraints?",
                "Before proceeding, let's ensure we have a clear understanding of the problem."
            ],
            "hints": "To better understand the problem, try summarizing the main objective and identifying key details or relationships."
        },
        "similar_problems": {
            "prompt": [
                "Have you encountered a problem similar to this before? How did you approach it?",
                "Are there any similarities between this problem and ones you've previously solved?",
                "Consider leveraging your past experiences to tackle the current problem."
            ],
            "hints": "Reflect on any strategies or techniques that proved effective in solving similar problems in the past."
        },
        "plan_approach": {
            "prompt": [
                "What strategies or approaches might be helpful in solving this problem?",
                "Let's brainstorm different methods we could use to approach this problem.",
                "Creating a plan of attack is essential. How can we break down the problem into manageable steps?"
            ],
            "hints": "Consider drawing diagrams, using formulas, or applying logical reasoning to develop your plan."
        },
        "carry_out_plan": {
            "prompt": [
                "Can you implement the plan you've devised? Are there any logical steps you can take to reach a solution?",
                "Let's execute the plan we've outlined. How can we systematically work through the problem?",
                "Carrying out the plan step by step is crucial. Let's proceed with confidence."
            ],
            "hints": "Focus on executing each step of your plan carefully and methodically, checking your progress along the way."
        },
        "look_back": {
            "prompt": [
                "Now that we've arrived at a solution, let's reflect on our approach. Does the solution make sense?",
                "Can you check your answer for accuracy and consistency? Does it align with the problem requirements?",
                "Looking back on our progress, are there any alternative solutions or approaches we could explore?"
            ],
            "hints": "Take time to review your solution critically, considering its validity and any alternative methods that may have been overlooked."
        }
    },
    "geometry": {
        "visualization": {
            "prompt": [
                "Can you visualize the shape or object described in the problem? What geometric properties are relevant?",
                "Drawing a diagram can often provide valuable insights. Let's sketch the given figure together.",
                "Visualization is key in geometry problems. Let's create a visual representation to aid our understanding."
            ],
            "hints": "Use diagrams or sketches to represent geometric figures accurately, focusing on relevant angles, lengths, and relationships."
        },
        "paper_cutting": {
            "prompt": [
                "For appropriate problems, consider using a paper-cutting approach. How can we physically manipulate the shape to gain insights?",
                "Let's try a hands-on approach. Can you cut out the shape from paper and experiment with different configurations?",
                "Paper cutting can offer a tangible perspective on geometric concepts. Let's explore its potential applications."
            ],
            "hints": "Experiment with cutting and folding paper to analyze geometric shapes, angles, and areas, observing how they change with manipulation."
        },
        "symmetry_congruence": {
            "prompt": [
                "Are there any symmetries or congruent parts in the shape? How can we leverage them to simplify the problem?",
                "Identifying symmetrical or congruent elements can streamline our approach. Let's search for patterns.",
                "Symmetry and congruence often reveal hidden relationships. Let's uncover their implications in this problem."
            ],
            "hints": "Look for symmetrical patterns or identical components within the geometric figure, as they may offer shortcuts or simplifications in problem-solving."
        }
    },
    "algebra": {
        "patterns_relationships": {
            "prompt": [
                "Can you identify any patterns or relationships in the numbers or equations provided?",
                "Analyzing patterns can provide valuable insights. Let's search for recurring elements or trends.",
                "Identifying algebraic relationships is key. Let's look for connections between variables and constants."
            ],
            "hints": "Look for recurring sequences, common factors, or consistent trends within the numerical or algebraic expressions, as they may indicate underlying relationships or properties."
        },
        "simplify_analyze": {
            "prompt": [
                "Is there an opportunity to simplify any expressions or equations before solving? How can we simplify the problem?",
                "Let's break down complex equations into smaller, more manageable components. How can we simplify our approach?",
                "Simplification is often the first step towards solving algebraic problems. Let's aim for clarity and efficiency."
            ],
            "hints": "Focus on simplifying expressions by factoring, expanding, or combining like terms, aiming to reduce complex equations to more digestible forms."
        },
        "estimation_approximation": {
            "prompt": [
                "Can you estimate the answer based on your understanding of the problem? How can estimation guide our approach?",
                "Estimation can provide valuable insights into the problem's solution. Let's explore approximation strategies.",
                "Using estimation as a tool can help us refine our solution. Let's leverage it to enhance our problem-solving process."
            ],
            "hints": "Use estimation techniques such as rounding, approximation, or order of magnitude analysis to gauge the reasonableness of your solution or to refine your approach."
        }
    },
    "calculus": {
        "rate_change": {
            "prompt": [
                "When dealing with functions, can you identify the rate of change involved? How does this information inform our approach?",
                "Understanding rates of change is essential in calculus problems. Let's analyze the function's behavior.",
                "Identifying rates of change can offer valuable insights into the problem's dynamics. Let's focus on this aspect."
            ],
            "hints": "Analyze the function's derivatives or slopes at critical points to determine the rate of change or the function's behavior over a given interval."
        },
        "limiting_values": {
            "prompt": [
                "Is the problem asking about a limit as a value approaches another? How can we visualize or manipulate the function to understand its behavior?",
                "Visualizing the function's behavior near certain values can provide insights. Let's explore its limiting behavior.",
                "Understanding limiting values is fundamental in calculus. Let's investigate how the function behaves under different conditions."
            ],
            "hints": "Use graphical or analytical methods to study the behavior of a function as it approaches certain values or as the independent variable approaches infinity or negative infinity."
        }
    },
    "proofs": {
        "hypothesis_conclusion": {
            "prompt": [
                "Can you clearly identify the hypothesis (what you're trying to prove) and the conclusion (what you need to demonstrate)?",
                "A clear statement of the hypothesis and conclusion is essential in proof-based problems. Let's define them precisely.",
                "Identifying the key elements of the proof statement is the first step. Let's clarify the hypothesis and conclusion."
            ],
            "hints": "Break down the proof statement into its constituent parts, clearly identifying the statement to be proven and the conditions or assumptions provided."
        },
        "indirect_proof": {
            "prompt": [
                "Can you prove the statement by assuming the opposite is true and leading to a contradiction?",
                "Indirect proof techniques can be powerful tools in proving statements. Let's explore this approach.",
                "Sometimes, proving by contradiction is more straightforward. Let's consider this alternative method."
            ],
            "hints": "Assume the negation of the statement to be proved and demonstrate that this assumption leads to a logical contradiction, thereby establishing the original statement's validity."
        }
    },
    "curiosity_creativity": {
        "what_if": {
            "prompt": [
                "What if we modify the problem slightly (change a parameter, add a condition)? How does this affect our approach?",
                "Exploring alternative scenarios can stimulate creativity. Let's consider different variations of the problem.",
                "What-if scenarios can lead to unexpected insights. Let's experiment with modifying the problem parameters."
            ],
            "hints": "Experiment with altering problem conditions, parameters, or constraints to explore how changes impact the problem's solution or approach."
        },
        "generalization": {
            "prompt": [
                "Can you generalize the problem to apply to a broader class of scenarios? How does this generalize our understanding?",
                "Generalizing the problem can reveal broader patterns or principles. Let's abstract the problem to a more general context.",
                "Generalization is a powerful tool in mathematics. Let's explore how the problem applies to other situations."
            ],
            "hints": "Identify commonalities or recurring themes within the problem and extend its principles to encompass a broader range of scenarios or contexts."
        },
        "real_world_applications": {
            "prompt": [
                "Can you think of a real-world situation where this mathematical concept might be applied?",
                "Connecting mathematical concepts to real-world applications can provide context and motivation. Let's explore potential applications.",
                "Understanding the real-world relevance of mathematical concepts is crucial. Let's investigate practical implications."
            ],
            "hints": "Consider how mathematical concepts or techniques can be used to model or solve real-world problems in fields such as science, engineering, finance, or economics."
        }
    },
    "metacognition": {
        "what_worked_well": {
            "prompt": [
                "After solving a problem, reflect on which strategies were most helpful. What techniques proved effective?",
                "Recognizing successful problem-solving strategies is key to improving future performance. Let's evaluate what worked well.",
                "Identifying successful approaches can inform our problem-solving toolkit. Let's celebrate our achievements."
            ],
            "hints": "Reflect on the strategies, techniques, or insights that contributed to your successful problem-solving process, considering how you can leverage them in future endeavors."
        },
        "what_didnt_work_well": {
            "prompt": [
                "Did you encounter any difficulties during the solving process? How can you adjust your approach for future challenges?",
                "Learning from setbacks is essential for growth. Let's analyze any challenges we encountered and consider adjustments.",
                "Understanding the limitations of our approach can guide future improvement efforts. Let's embrace the learning opportunities."
            ],
            "hints": "Identify any obstacles, challenges, or areas of difficulty encountered during the problem-solving process, reflecting on strategies to overcome or mitigate them in future endeavors."
        }
    },
    "bonus_prompt": {
        "explore_further": {
            "prompt": [
                "Did you find this problem interesting? Are there any related concepts you'd like to learn more about?",
                "Exploring additional topics or related concepts can deepen your understanding. Let's delve into related areas of interest.",
                "Learning doesn't stop with solving a single problem. Let's continue our exploration and expand our mathematical horizons."
            ],
            "hints": "Identify aspects of the problem that pique your curiosity or areas of related mathematics that you find intriguing, exploring opportunities for further investigation and learning."
        }
    }
}
