def majority_vote(predictions: list[int]) -> int:
    return max(set(predictions) , key=predictions.count)

def weighted_vote(predictions: list[int] , weights : list[float])-> int:
    total = sum(w for w in weights)
    weighted_sum = sum(p * w for p , w in zip(predictions , weights))

    return 1 if weighted_sum > 0.5 else 0