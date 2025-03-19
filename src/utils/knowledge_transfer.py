"""
Knowledge Transfer Manager for Collaborative Poker AI Learning
"""

import random
import logging
from typing import Dict, Any, List


class KnowledgeTransferManager:
    """
    Manages knowledge transfer between AI models in the poker learning ecosystem.
    """

    def __init__(self,
                 transfer_rate: float = 0.5,
                 similarity_threshold: float = 0.7):
        """
        Initialize knowledge transfer manager.

        Args:
            transfer_rate: Probability of knowledge transfer
            similarity_threshold: Minimum similarity for knowledge transfer
        """
        self.transfer_rate = transfer_rate
        self.similarity_threshold = similarity_threshold

        # Logging setup
        self.logger = logging.getLogger(__name__)

        # Knowledge transfer tracking
        self.transfer_history = []
        self.global_knowledge_pool = {}

    def transfer_knowledge(self, source_model, target_model):
        """
        Transfer knowledge from source to target model.

        Args:
            source_model: Model providing knowledge
            target_model: Model receiving knowledge
        """
        # Check transfer probability
        if random.random() > self.transfer_rate:
            return

        # Extract knowledge from source model
        source_knowledge = self._extract_model_knowledge(source_model)

        # Compute knowledge similarity
        similarity = self._compute_knowledge_similarity(
            source_knowledge,
            self._extract_model_knowledge(target_model)
        )

        # Transfer if similarity is above threshold
        if similarity >= self.similarity_threshold:
            target_model.integrate_external_knowledge(source_knowledge)

            # Update transfer history
            self._log_knowledge_transfer(source_model, target_model, similarity)

    def _extract_model_knowledge(self, model) -> Dict[str, Any]:
        """
        Extract key knowledge components from a model.

        Args:
            model: Model to extract knowledge from

        Returns:
            Dictionary of extracted knowledge
        """
        # Extract key state information and regrets
        state_knowledge = {}
        for state_hash, regrets in model.cumulative_regrets.items():
            # Select top actions with highest absolute regret
            top_actions = sorted(
                regrets.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 most significant actions

            state_knowledge[state_hash] = {
                'regrets': dict(top_actions),
                'total_regret': sum(abs(r) for r in regrets.values())
            }

        return {
            'contribution_value': model.collaborative_score,
            'state_knowledge': state_knowledge,
            'hyperparameters': {
                'discount_factor': model.discount_factor,
                'learning_rate': model.learning_rate,
                'exploration_factor': model.exploration_factor
            }
        }

    def _compute_knowledge_similarity(self,
                                      knowledge1: Dict[str, Any],
                                      knowledge2: Dict[str, Any]) -> float:
        """
        Compute similarity between two knowledge representations.

        Args:
            knowledge1: First knowledge dictionary
            knowledge2: Second knowledge dictionary

        Returns:
            Similarity score (0-1)
        """
        # Compare state knowledge and hyperparameters
        state_similarity = self._compare_state_knowledge(
            knowledge1.get('state_knowledge', {}),
            knowledge2.get('state_knowledge', {})
        )

        # Compare hyperparameters
        param_similarity = self._compare_hyperparameters(
            knowledge1.get('hyperparameters', {}),
            knowledge2.get('hyperparameters', {})
        )

        # Weighted combination of similarities
        return 0.7 * state_similarity + 0.3 * param_similarity

    def _compare_state_knowledge(self,
                                 states1: Dict[str, Any],
                                 states2: Dict[str, Any]) -> float:
        """
        Compare state knowledge between two models.

        Args:
            states1: First model's state knowledge
            states2: Second model's state knowledge

        Returns:
            State knowledge similarity score
        """
        # Find common state hashes
        common_states = set(states1.keys()) & set(states2.keys())

        if not common_states:
            returnreturn 0.0  # No common states

        # Compute state knowledge similarity
        state_similarities = []
        for state_hash in common_states:
            # Compare regrets for top actions
            state1 = states1[state_hash]
            state2 = states2[state_hash]

            # Compare action regrets
            common_actions = set(state1['regrets'].keys()) & set(state2['regrets'].keys())

            if common_actions:
                action_similarities = []
                for action in common_actions:
                    # Normalized regret comparison
                    max_regret = max(
                        abs(state1['regrets'][action]),
                        abs(state2['regrets'][action])
                    )
                    similarity = 1 - abs(
                        state1['regrets'][action] - state2['regrets'][action]
                    ) / (max_regret + 1e-10)
                    action_similarities.append(similarity)

                # Average action similarities for this state
                state_similarities.append(
                    sum(action_similarities) / len(action_similarities)
                )

        # Return average state knowledge similarity
        return sum(state_similarities) / len(state_similarities) if state_similarities else 0.0

    def _compare_hyperparameters(self,
                                 params1: Dict[str, float],
                                 params2: Dict[str, float]) -> float:
        """
        Compare hyperparameters between two models.

        Args:
            params1: First model's hyperparameters
            params2: Second model's hyperparameters

        Returns:
            Hyperparameter similarity score
        """
        if not params1 or not params2:
            return 0.0

        # Define parameter weights
        param_weights = {
            'discount_factor': 0.4,
            'learning_rate': 0.3,
            'exploration_factor': 0.3
        }

        # Compute weighted similarity
        similarities = []
        for param, weight in param_weights.items():
            if param in params1 and param in params2:
                # Compute parameter similarity using gaussian similarity
                max_val = max(abs(params1[param]), abs(params2[param]))
                similarity = 1 - abs(params1[param] - params2[param]) / (max_val + 1e-10)
                similarities.append(similarity * weight)

        return sum(similarities)

    def _log_knowledge_transfer(self,
                                source_model,
                                target_model,
                                similarity: float):
        """
        Log knowledge transfer event.

        Args:
            source_model: Model providing knowledge
            target_model: Model receiving knowledge
            similarity: Knowledge similarity score
        """
        transfer_event = {
            'timestamp': random.random(),  # Placeholder for actual timestamp
            'source_iteration': source_model.iteration_count,
            'target_iteration': target_model.iteration_count,
            'similarity': similarity,
            'source_collaborative_score': source_model.collaborative_score,
            'target_collaborative_score': target_model.collaborative_score
        }

        # Store transfer event
        self.transfer_history.append(transfer_event)

        # Update global knowledge pool
        self._update_global_knowledge_pool(transfer_event)

        # Log transfer details
        self.logger.info(
            f"Knowledge Transfer: "
            f"Similarity={similarity:.2f}, "
            f"Source Iter={transfer_event['source_iteration']}, "
            f"Target Iter={transfer_event['target_iteration']}"
        )

    def _update_global_knowledge_pool(self, transfer_event: Dict[str, Any]):
        """
        Update the global knowledge pool with transfer insights.

        Args:
            transfer_event: Details of knowledge transfer
        """
        # Aggregate transfer insights
        for key, value in transfer_event.items():
            if key not in self.global_knowledge_pool:
                self.global_knowledge_pool[key] = []

            self.global_knowledge_pool[key].append(value)

        # Prune old knowledge if pool gets too large
        max_pool_size = 1000
        for key in self.global_knowledge_pool:
            if len(self.global_knowledge_pool[key]) > max_pool_size:
                self.global_knowledge_pool[key] = self.global_knowledge_pool[key][-max_pool_size:]

    def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive knowledge transfer statistics.

        Returns:
            Dictionary of transfer insights
        """
        if not self.transfer_history:
            return {}

        # Compute transfer statistics
        stats = {
            'total_transfers': len(self.transfer_history),
            'avg_similarity': sum(
                event['similarity'] for event in self.transfer_history
            ) / len(self.transfer_history),
            'max_similarity': max(
                event['similarity'] for event in self.transfer_history
            ),
            'iteration_range': {
                'min_source_iteration': min(
                    event['source_iteration'] for event in self.transfer_history
                ),
                'max_source_iteration': max(
                    event['source_iteration'] for event in self.transfer_history
                )
            },
            'collaborative_score_impact': self._analyze_collaborative_score_impact()
        }

        return stats

    def _analyze_collaborative_score_impact(self) -> Dict[str, float]:
        """
        Analyze the impact of knowledge transfer on collaborative scores.

        Returns:
            Dictionary of collaborative score insights
        """
        if not self.transfer_history:
            return {}

        # Compute score changes
        source_score_changes = [
            event['source_collaborative_score'] for event in self.transfer_history
        ]
        target_score_changes = [
            event['target_collaborative_score'] for event in self.transfer_history
        ]

        return {
            'avg_source_score_before_transfer': np.mean(source_score_changes),
            'avg_target_score_after_transfer': np.mean(target_score_changes),
            'score_improvement_ratio': np.mean(
                [t / (s + 1e-10) for s, t in zip(source_score_changes, target_score_changes)]
            )
        }

    def reset_transfer_history(self):
    """
    Reset knowledge transfer history and global knowledge pool.
    """
    self.transfer_history = []
    self.global_knowledge_pool = {}
    self.logger.info("Knowledge transfer history and global knowledge pool reset.")