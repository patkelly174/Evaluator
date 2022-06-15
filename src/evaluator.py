import math
import matplotlib.pyplot as plt

# Patrick Kelly

class Evaluator:

    def find_intersection(self, list1, list2):
        final = [value for value in list1 if value in list2]
        return final

    def get_relevance_given_doc(self, list, document):
        for element in list:
            if element[0] == document:
                return element[1]
        return 0

    def get_all_relevant_docs(self, list):
        final = []
        for element in list:
            if element[1] > 0:
                final.append(element[0])
        return final

    def get_docs(self, pq, k):
        final = []
        rank = 0
        while rank < k:
            try:
                final.append(pq[rank][0])
            except IndexError:
                break
            rank += 1
        return final

    # NDCG@K
    def ndcg_at_k(self, qrels, arr, k):
        num = 0
        sum = 0
        for element in arr:
            rank = 1
            score = 0
            doc = arr[element][rank-1][0]
            rank += 1
            init_rel = self.get_relevance_given_doc(qrels[element], doc)
            ideal_dcg = sorted(qrels[element], key = lambda x: x[1], reverse = True)
            score += init_rel / ideal_dcg[0][1]
            while rank <= k:
                try:
                    document = arr[element][rank-1][0]
                except IndexError:
                    break
                rel_i = self.get_relevance_given_doc(qrels[element], document)
                dcg = rel_i / math.log(rank, 2)
                ideal = ideal_dcg[rank - 1][1] / math.log(rank, 2)
                if ideal <= 0:
                    score += 0
                else:
                    score += dcg / ideal
                rank += 1
            num += 1
            sum += score
        return sum / num

    # MRR
    def mrr(self, qrels, arr):
        num = 0
        sum = 0
        for element in arr:
            rank = 0
            rel = 0
            while not rel > 0:
                try:
                    document = arr[element][rank][0]
                except IndexError:
                    break
                rank += 1
                rel = self.get_relevance_given_doc(qrels[element], document)
            if rel > 0:
                score = 1 / rank
            else:
                score = 0
            num += 1
            sum += score
        return sum / num

    # P@K
    def precision_at_k(self, qrels, arr, k):
        num = 0
        sum = 0
        for element in arr:
            retrieved_docs = self.get_docs(arr[element], k)
            relevant_docs = self.get_all_relevant_docs(qrels[element])
            intersect_length = len(self.find_intersection(relevant_docs, retrieved_docs))
            length = len(retrieved_docs)
            if length > 0:
                sum += intersect_length / length
            num += 1
        return sum / num

    # Recall@K
    def recall_at_k(self, qrels, arr, k):
        sum = 0
        num = 0
        for element in arr:
            retrieved_docs = self.get_docs(arr[element], k)
            relevant_docs = self.get_all_relevant_docs(qrels[element])
            intersect_length = len(self.find_intersection(relevant_docs, retrieved_docs))
            length = len(relevant_docs)
            if length > 0:
                sum += intersect_length / length
            num += 1
        return sum / num

    # MAP
    def mean_at_p(self, qrels, arr):
        sum = 0
        num = 0
        for element in arr:
            relevant_docs_length = len(self.get_all_relevant_docs(qrels[element]))
            count = 0
            ap = 0
            rel_seen = 0
            while True:
                try:
                    document = arr[element][count][0]
                except IndexError:
                    break
                rel = self.get_relevance_given_doc(qrels[element], document)
                if rel > 0:
                    rel_seen += 1
                    ap += rel_seen / (count + 1)
                count += 1
            if relevant_docs_length > 0:
                ap = ap / relevant_docs_length
            else:
                ap = 0
            sum += ap
            num += 1
        return sum / num

    # F1@K
    def f1_at_k(self, qrels, arr, k):
        precision = self.precision_at_k(qrels, arr, k)
        recall = self.recall_at_k(qrels, arr, k)
        if not (precision + recall) > 0:
            result = 0
        else:
            result = (2 * precision * recall) / (precision + recall)
        return result

if __name__ == "__main__":
    qrels = {}
    bm25 = {}
    ql = {}
    sdm = {}
    stress = {}

    def file_helper(file_name):
        file = open(file_name, "r")
        arr = {}
        for line in file:
            list = line.split()
            element = list[0]
            document = list[2]
            rel = int(list[3])
            if element not in arr:
                arr[element] = [(document, rel)]
            else:
                arr[element].append((document, rel))
        file.close()
        return arr

    qrels = file_helper("evaluation-data/qrels")
    bm25 = file_helper("evaluation-data/bm25.trecrun")
    ql = file_helper("evaluation-data/ql.trecrun")
    sdm = file_helper("evaluation-data/sdm.trecrun")
    stress = file_helper("evaluation-data/stress.trecrun")

    evaluator = Evaluator()
    calls = [
        "bm25.trecrun NDCG@15 " + str(evaluator.ndcg_at_k(qrels, bm25, 15)) + "\n",
        "bm25.trecrun MRR " + str(evaluator.mrr(qrels, bm25)) + "\n",
        "bm25.trecrun P@5 " + str(evaluator.precision_at_k(qrels, bm25, 5)) + "\n",
        "bm25.trecrun P@10 " + str(evaluator.precision_at_k(qrels, bm25, 10)) + "\n",
        "bm25.trecrun Recall@10 " + str(evaluator.recall_at_k(qrels, bm25, 10)) + "\n",
        "bm25.trecrun F1@10 " + str(evaluator.f1_at_k(qrels, bm25, 10)) + "\n",
        "bm25.trecrun MAP " + str(evaluator.mean_at_p(qrels, bm25)) + "\n\n",
        "ql.trecrun NDCG@15 " + str(evaluator.ndcg_at_k(qrels, ql, 15)) + "\n",
        "ql.trecrun MRR " + str(evaluator.mrr(qrels, ql)) + "\n",
        "ql.trecrun P@5 " + str(evaluator.precision_at_k(qrels, ql, 5)) + "\n",
        "ql.trecrun P@10 " + str(evaluator.precision_at_k(qrels, ql, 10)) + "\n",
        "ql.trecrun Recall@10 " + str(evaluator.recall_at_k(qrels, ql, 10)) + "\n",
        "ql.trecrun F1@10 " + str(evaluator.f1_at_k(qrels, ql, 10)) + "\n",
        "ql.trecrun MAP " + str(evaluator.mean_at_p(qrels, ql)) + "\n\n",
        "sdm.trecrun NDCG@15 " + str(evaluator.ndcg_at_k(qrels, sdm, 15)) + "\n",
        "sdm.trecrun MRR " + str(evaluator.mrr(qrels, sdm)) + "\n",
        "sdm.trecrun P@5 " + str(evaluator.precision_at_k(qrels, sdm, 5)) + "\n",
        "sdm.trecrun P@10 " + str(evaluator.precision_at_k(qrels, sdm, 10)) + "\n",
        "sdm.trecrun Recall@10 " + str(evaluator.recall_at_k(qrels, sdm, 10)) + "\n",
        "sdm.trecrun F1@10 " + str(evaluator.f1_at_k(qrels, sdm, 10)) + "\n",
        "sdm.trecrun MAP " + str(evaluator.mean_at_p(qrels, sdm)) + "\n\n",
        "stress.trecrun NDCG@15 " + str(evaluator.ndcg_at_k(qrels, stress, 15)) + "\n",
        "stress.trecrun MRR " + str(evaluator.mrr(qrels, stress)) + "\n",
        "stress.trecrun P@5 " + str(evaluator.precision_at_k(qrels, stress, 5)) + "\n",
        "stress.trecrun P@10 " + str(evaluator.precision_at_k(qrels, stress, 10)) + "\n",
        "stress.trecrun Recall@10 " + str(evaluator.recall_at_k(qrels, stress, 10)) + "\n",
        "stress.trecrun F1@10 " + str(evaluator.f1_at_k(qrels, stress, 10)) + "\n",
        "stress.trecrun MAP " + str(evaluator.mean_at_p(qrels, stress))
    ]


    element450_bm25 = {'450': bm25['450']}
    element450_ql = {'450': ql['450']}
    element450_sdm = {'450': sdm['450']}

    precision_bm25 = []
    recall_bm25 = []
    precision_ql = []
    recall_ql = []
    precision_sdm = []
    recall_sdm = []

    for element in range(1, len(ql['450'])):
        recall_ql.append(evaluator.recall_at_k(qrels, element450_ql, element))
        precision_ql.append(evaluator.precision_at_k(qrels, element450_ql, element))

    for element in range(1, len(ql['450'])):
        recall_bm25.append(evaluator.recall_at_k(qrels, element450_bm25, element))
        precision_bm25.append(evaluator.precision_at_k(qrels, element450_bm25, element))

    for element in range(1, len(sdm['450'])):
        recall_sdm.append(evaluator.recall_at_k(qrels, element450_sdm, element))
        precision_sdm.append(evaluator.precision_at_k(qrels, element450_sdm, element))

    plt.title("Recall v Precision")

    plt.plot(recall_sdm, precision_sdm, label="SDM")
    plt.plot(recall_bm25, precision_bm25, label="BM25")
    plt.plot(recall_ql, precision_ql, label="QL")
    plt.legend()
    plt.show()

    plt.plot(recall_bm25, precision_bm25)
    plt.title("Recall v Precision of element 450 w/ BM25")
    plt.show()

    output = open("output.metrics", 'w')
    for element in calls:
        output.write(element)
    output.close()
