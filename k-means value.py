from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeature,
    QgsField,
    QgsWkbTypes
)
from qgis.PyQt.QtCore import QVariant
import math

class BalancedKMeans(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    CLUSTERS = "CLUSTERS"
    MINPOINTS = "MINPOINTS"
    MAXITER = "MAXITER"
    OUTPUT = "OUTPUT"

    # --- metadata required oleh QGIS ---
    def name(self):
        return "balancedkmeans_hardmin"

    def displayName(self):
        return "Balanced KMeans (Hard min points + Value balance)"

    def group(self):
        return "Clustering"

    def groupId(self):
        return "clustering"

    def shortHelpString(self):
        return ("Bagi titik menjadi k cluster dengan syarat setiap cluster minimal memiliki "
                "min_points (hard). Setelah min terpenuhi, sisa poin dialokasikan untuk "
                "meminimalkan selisih total value antar cluster. Refinement swap tidak boleh "
                "mengurangi cluster di bawah min_points.")

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT, "Input point layer", [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD, "Value field", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLUSTERS, "Number of clusters (k)",
                QgsProcessingParameterNumber.Integer, minValue=2, defaultValue=3
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MINPOINTS, "Minimum points per cluster (hard constraint)",
                QgsProcessingParameterNumber.Integer, minValue=1, defaultValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXITER, "Max iterations (refinement swaps)",
                QgsProcessingParameterNumber.Integer, minValue=1, defaultValue=50
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, "Clustered output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        field_name = self.parameterAsString(parameters, self.FIELD, context)
        k = int(self.parameterAsInt(parameters, self.CLUSTERS, context))
        min_pts = int(self.parameterAsInt(parameters, self.MINPOINTS, context))
        max_iter = int(self.parameterAsInt(parameters, self.MAXITER, context))

        # collect features
        features = list(source.getFeatures())
        n = len(features)
        if n == 0:
            raise Exception("Layer kosong.")

        # validate feasibility
        if k * min_pts > n:
            raise Exception(f"Tidak cukup titik: {n} titik tidak bisa dipenuhi {k} cluster x {min_pts} min.")

        # coords & values arrays (preserve feature order)
        coords = []
        values = []
        for f in features:
            geom = f.geometry()
            if geom.isEmpty():
                coords.append((0.0, 0.0))
            else:
                p = geom.asPoint()
                coords.append((p.x(), p.y()))
            # safe numeric parsing
            try:
                v = float(f[field_name]) if f[field_name] is not None else 0.0
            except Exception:
                v = 0.0
            values.append(v)

        total_value = sum(values)
        target_value = total_value / k if k > 0 else 0.0

        # spatial normalization
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bbox_dx = max(xs) - min(xs) if xs else 1.0
        bbox_dy = max(ys) - min(ys) if ys else 1.0
        spatial_scale = math.hypot(bbox_dx, bbox_dy) or 1.0

        def dist(i, j):
            return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

        def distance_to_centroid(i, centroid):
            return math.hypot(coords[i][0]-centroid[0], coords[i][1]-centroid[1])

        # init centroids as k-means++ style (farthest-point)
        centroids = [coords[0]]
        for _ in range(1, k):
            dists = [min(math.hypot(pt[0]-c[0], pt[1]-c[1]) for c in centroids) for pt in coords]
            idx = max(range(len(dists)), key=lambda ii: dists[ii])
            centroids.append(coords[idx])

        # labels & cluster aggregates
        labels = [-1] * n
        cluster_sums = [0.0] * k
        cluster_counts = [0] * k

        # order points by value descending
        order = sorted(range(n), key=lambda i: values[i], reverse=True)

        # STEP 1: allocate min_pts per cluster using top values in round-robin
        p = 0
        for r in range(min_pts):
            for c in range(k):
                idx = order[p]
                labels[idx] = c
                cluster_sums[c] += values[idx]
                cluster_counts[c] += 1
                p += 1

        # STEP 2: allocate remaining points to minimize value imbalance (with distance tie-break)
        for idx in order[p:]:
            best_c = None
            best_score = None
            for c in range(k):
                # compute how far new sum would be from target (relative)
                new_sum = cluster_sums[c] + values[idx]
                value_term = abs(new_sum - target_value) / (target_value if target_value != 0 else 1.0)
                # distance to centroid (normalize)
                d = distance_to_centroid(idx, centroids[c]) / spatial_scale
                # combined score: give stronger weight to value balance; distance as tie-break
                score = value_term + 0.2 * d
                if best_score is None or score < best_score:
                    best_score = score
                    best_c = c
            labels[idx] = best_c
            cluster_sums[best_c] += values[idx]
            cluster_counts[best_c] += 1

        # recompute centroids (average point)
        for c in range(k):
            pts = [i for i, lab in enumerate(labels) if lab == c]
            if pts:
                sx = sum(coords[i][0] for i in pts)
                sy = sum(coords[i][1] for i in pts)
                centroids[c] = (sx / len(pts), sy / len(pts))

        # STEP 3: refinement swaps to reduce global imbalance while preserving min_pts
        def total_imbalance(sums):
            return sum(abs(s - target_value) for s in sums)

        for it in range(max_iter):
            improved = False
            current_imb = total_imbalance(cluster_sums)
            # try moving each point to better cluster if reduces imbalance and preserves min_pts
            for i in range(n):
                cur = labels[i]
                for c in range(k):
                    if c == cur:
                        continue
                    # if moving would break min_pts for current cluster, skip
                    if cluster_counts[cur] - 1 < min_pts:
                        continue
                    new_sums = cluster_sums.copy()
                    new_sums[cur] -= values[i]
                    new_sums[c] += values[i]
                    new_imb = total_imbalance(new_sums)
                    # accept move only if global imbalance reduces
                    if new_imb < current_imb - 1e-6:
                        # (optional) also prefer moves that do not create huge spatial jump:
                        # allow move if reasonable (we already prioritized value)
                        labels[i] = c
                        cluster_sums[cur] -= values[i]
                        cluster_counts[cur] -= 1
                        cluster_sums[c] += values[i]
                        cluster_counts[c] += 1
                        # update centroids of affected clusters
                        for cc in (cur, c):
                            pts = [ii for ii, lab in enumerate(labels) if lab == cc]
                            if pts:
                                sx = sum(coords[ii][0] for ii in pts)
                                sy = sum(coords[ii][1] for ii in pts)
                                centroids[cc] = (sx / len(pts), sy / len(pts))
                        improved = True
                        current_imb = new_imb
                        break
                if improved:
                    # continue outer loop to attempt more improvements
                    pass
            if not improved:
                break

        # final aggregates (ensure all clusters valid)
        final_sums = [0.0]*k
        final_counts = [0]*k
        for i, lab in enumerate(labels):
            final_sums[lab] += values[i]
            final_counts[lab] += 1

        # prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("cluster_id", QVariant.Int))
        out_fields.append(QgsField("cluster_sum", QVariant.Double))
        out_fields.append(QgsField("cluster_pct", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
        )

        # write output features
        for f, lab in zip(features, labels):
            new_f = QgsFeature(out_fields)
            new_f.setGeometry(f.geometry())
            attrs = f.attributes()
            # cluster id starting from 1
            attrs.append(int(lab) + 1)
            cs = float(final_sums[lab])
            attrs.append(cs)
            pct = (cs / total_value * 100.0) if total_value > 0 else 0.0
            attrs.append(pct)
            new_f.setAttributes(attrs)
            sink.addFeature(new_f)

        # feedback log
        feedback.pushInfo(f"Total points: {n}; clusters: {k}; min_pts per cluster: {min_pts}")
        feedback.pushInfo(f"Target value per cluster: {target_value:.2f}")
        for cid in range(k):
            feedback.pushInfo(f"Cluster {cid+1}: sum={final_sums[cid]:.2f}, count={final_counts[cid]}")

        return {self.OUTPUT: dest_id}

    def createInstance(self):
        return BalancedKMeans()
