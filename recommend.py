import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--artistid', help="(artist id) see: musicdata/artists.dat")
args = parser.parse_args()

def loadArtistNames():
    art = {}
    with open("musicdata/artists.dat") as f:
        for line in f:
            fields = line.split('\t')
            art[int(fields[0])] = fields[1]
    return art

def makePairs( listens ):
    ratings = listens[1]
    (art1, list1) = ratings[0]
    (art2, list2) = ratings[1]
    return ((art1, art2), (list1, list2))

def filterDuplicates( listens ):
    ratings = listens[1]
    (art1, list1) = ratings[0]
    (art2, list2) = ratings[1]
    return art1 < art2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

conf = SparkConf().setMaster("local[*]").setAppName("ArtistSimilarities")
sc = SparkContext(conf = conf)

print("\nLoading artist names...")
nameDict = loadArtistNames()

data = sc.textFile("musicdata/user_artists.dat")

ratings = data.map(lambda l: l.split('\t')).map(lambda l: (int(l[0]), (int(l[1]), int(l[2]))))
joinedRatings = ratings.join(ratings)
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
artistPairs = uniqueJoinedRatings.map(makePairs)
artistPairRatings = artistPairs.groupByKey()
artistPairSimilarities = artistPairRatings.mapValues(computeCosineSimilarity).cache()

if (args.artistid):

    scoreThreshold = 0.75
    coOccurenceThreshold = 8

    artistID = int(args.artistid)

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = artistPairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == artistID or pairSim[0][1] == artistID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar artists for " + nameDict[artistID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarartistID = pair[0]
        if (similarartistID == artistID):
            similarartistID = pair[1]
        print(nameDict[similarartistID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
