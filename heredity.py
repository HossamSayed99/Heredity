import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Getting Ancestors (i.e : people who have no parents)
    parents = []
    for person in people:
        if people[person]["mother"] == None:
            parents.append(people[person]['name'])
            
    values_computed = dict()
    
    for person in people.keys():
        values_computed.update({people[person]['name']: []})    

    # First I compute the value for parents
    for parent in parents:
        if parent in one_gene:
            # I compute that the parent has one gene
            values_computed[parent].append(PROBS["gene"][1])

            # I compute that a parent does or doesnot have the trait given it has one gene
            if parent in have_trait:
                values_computed[parent].append(PROBS["trait"][1][True])
            else:
                values_computed[parent].append(PROBS["trait"][1][False])

        elif parent in two_genes:
            # I compute that the parent has two genes
            values_computed[parent].append(PROBS["gene"][2])

            # I compute that a parent does or doesnot have the trait given it has two genes
            if parent in have_trait:
                values_computed[parent].append(PROBS["trait"][2][True])
            else:
                values_computed[parent].append(PROBS["trait"][2][False])

        else:
            # I compute that the parent has no genes
            values_computed[parent].append(PROBS["gene"][0])
            # I compute that a parent does or doesnot have the trait given it has no genes
            if parent in have_trait:
                values_computed[parent].append(PROBS["trait"][0][True])
            else:
                values_computed[parent].append(PROBS["trait"][0][False])
           
    children = [people[person]['name'] for person in people.keys() if people[person]['name'] not in parents]    
    
    for child in children:
        # Computing the probability that the child gets the gene from his mother or / and his father 

        # If the mother has zero genes, one gene or two genes
        if people[child]['mother'] not in one_gene and people[child]['mother'] not in two_genes:
            chance_to_not_get_from_mother = 1 - PROBS['mutation']
            chance_to_get_from_mother = PROBS['mutation']

        elif people[child]['mother'] in one_gene:
            chance_to_not_get_from_mother = 0.5
            chance_to_get_from_mother = 0.5
        else:
            chance_to_not_get_from_mother = PROBS['mutation']
            chance_to_get_from_mother = 1 - PROBS['mutation']

        # If the father has zero genes, one gene or two genes
        if people[child]['father'] not in one_gene and people[child]['father'] not in two_genes:
            chance_to_not_get_from_father = 1 - PROBS['mutation']
            chance_to_get_from_father = PROBS['mutation']

        elif people[child]['father'] in one_gene:
            chance_to_not_get_from_father = 0.5
            chance_to_get_from_father = 0.5

        else:
            chance_to_not_get_from_father = PROBS['mutation']
            chance_to_get_from_father = 1 - PROBS['mutation']
        
        # Computing the probability for the child given the state of the parent

        if child not in one_gene and child not in two_genes:
            values_computed[child].append(chance_to_not_get_from_mother * chance_to_not_get_from_father)

            # I compute that a child does or doesnot have the trait given it has no genes
            if child in have_trait:
                values_computed[child].append(PROBS["trait"][0][True])
            else:
                values_computed[child].append(PROBS["trait"][0][False])

        elif child in one_gene:
            values_computed[child].append(chance_to_not_get_from_mother * chance_to_get_from_father +
                                          chance_to_get_from_mother * chance_to_not_get_from_father)

            # I compute that a child does or doesnot have the trait given it has one gene
            if child in have_trait:
                values_computed[child].append(PROBS["trait"][1][True])
            else:
                values_computed[child].append(PROBS["trait"][1][False])
        else:

            values_computed[child].append(chance_to_get_from_mother * chance_to_get_from_father)

            # I compute that a child does or doesnot have the trait given it has two genes
            if child in have_trait:
                values_computed[child].append(PROBS["trait"][2][True])
            else:
                values_computed[child].append(PROBS["trait"][2][False])
            
    # print(values_computed)
    # Multiplying all values by each other
    ret = 1

    for person in values_computed:
        for val in values_computed[person]:
            ret *= val

    return ret


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Adding p to the respective probabilities
    for person in probabilities:

        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p
    
    return 


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Computing the Corerction factor which is equal to the 1 / total values and then multiplying each value by this factor
    for person in probabilities:
        
        total_gene_values = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        correctionFactor = 1 / total_gene_values
        probabilities[person]["gene"][0] *= correctionFactor
        probabilities[person]["gene"][1] *= correctionFactor
        probabilities[person]["gene"][2] *= correctionFactor

        total_trait_values = probabilities[person]["trait"][False] + probabilities[person]["trait"][True]
        correctionFactor = 1 / total_trait_values
        probabilities[person]["trait"][False] *= correctionFactor
        probabilities[person]["trait"][True] *= correctionFactor

    return


if __name__ == "__main__":
    main()
