package io.blanco

import java.util.Arrays

import breeze.linalg.qr.reduced.{justQ => bjq}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, LU => BLU, SparseVector => BSV, axpy => brzAxpy, qr => BQR, svd => brzSvd}
import breeze.math.Semiring
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrices, SingularValueDecomposition, Vectors, Matrix => SparkMatrix}

object fastSVD {
  /**
    * Computes singular value decomposition of this matrix. Denote this matrix by A (m x n). This
    * will compute matrices U, S, V such that A ~= U * S * V', where S contains the leading k
    * singular values, U and V contain the corresponding singular vectors.
    *
    * At most k largest non-zero singular values and associated vectors are returned. If there are k
    * such values, then the dimensions of the return will be:
    *  - U is a RowMatrix of size m x k that satisfies U' * U = eye(k),
    *  - s is a Vector of size k, holding the singular values in descending order,
    *  - V is a Matrix of size n x k that satisfies V' * V = eye(k).
    *
    * We assume n is smaller than m, though this is not strictly required.
    * The singular values and the right singular vectors are derived
    * from the eigenvalues and the eigenvectors of the Gramian matrix A' * A. U, the matrix
    * storing the right singular vectors, is computed via matrix multiplication as
    * U = A * (V * S^-1^), if requested by user. The actual method to use is determined
    * automatically based on the cost:
    *  - If n is small (n &lt; 100) or k is large compared with n (k &gt; n / 2), we compute
    * the Gramian matrix first and then compute its top eigenvalues and eigenvectors locally
    * on the driver. This requires a single pass with O(n^2^) storage on each executor and
    * on the driver, and O(n^2^ k) time on the driver.
    *  - Otherwise, we compute (A' * A) * v in a distributive way and send it to ARPACK's DSAUPD to
    * compute (A' * A)'s top eigenvalues and eigenvectors on the driver node. This requires O(k)
    * passes, O(n) storage on each executor, and O(n k) storage on the driver.
    *
    * Several internal parameters are set to default values. The reciprocal condition number rCond
    * is set to 1e-9. All singular values smaller than rCond * sigma(0) are treated as zeros, where
    * sigma(0) is the largest singular value. The maximum number of Arnoldi update iterations for
    * ARPACK is set to 300 or k * 3, whichever is larger. The numerical tolerance for ARPACK's
    * eigen-decomposition is set to 1e-10.
    *
    * @param a RowMatrix to perform SVD
    * @param k number of leading singular values to keep (0 &lt; k &lt;= n).
    *          It might return less than k if
    *          there are numerically zero singular values or there are not enough Ritz values
    *          converged before the maximum number of Arnoldi update iterations is reached (in case
    *          that matrix A is ill-conditioned).
    *          are treated as zero, where sigma(0) is the largest singular value.
    * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
    * @note The conditions that decide which method to use internally and the default parameters are
    *       subject to change.
    */

  /**
    * The actual SVD implementation, visible for testing.
    *
    * @param a        The matrix to compute SVD on
    * @param k        number of leading singular values to keep (0 &lt; k &lt;= n)
    * @param center   Whether or not the data must be centered
    *                 (if center = true rows will be mean-centered)
    * @param p_iter   The number of power iterations to conduct. defaults to 2
    * @param blk_size The block size of the normalized power iterations, defaults to k+2
    * @return SingularValueDecomposition(U, s, V)
    */
  def apply(
             a: RowMatrix,
             k: Int,
             center: Boolean = false,
             p_iter: Int = 2,
             blk_size: Int = -1): SingularValueDecomposition[SparkMatrix, SparkMatrix] = {

    var n = a.numCols().toInt
    require(k > 0 && k <= n, s"Requested k singular values but got k=$k and numCols=$n.")

    var bs: Int = 0;
    if (blk_size == -1) {
      bs = k + 2
    }
    if (bs < 0) {
      throw new IllegalArgumentException("Block size for SVD must be > 0, defaults to k+2")
    }
    val m: Long = a.numRows()
    val nc = a.numCols()
    val maxK = min(m, nc)
    require(k <= maxK, "number of singular values must be less than min(rows, cols)")
    var c: BDM[Double] = null
    if (center) {
      // TODO: Center the 'a' matrix
      // *_Technically_* not finished
      // Center the matrix first to get PCA results
      c = RowMatrixToBreeze(a)
    }
    c = RowMatrixToBreeze(a)

    // Use the val "c" from here to refer to the source data matrix
    if (bs >= m / 1.25 || bs >= n / 1.25) {
      // Perform NORMAL SVD Here.
      // Return the SVD from here
      var lameSVD: SingularValueDecomposition[RowMatrix, SparkMatrix] = a.computeSVD(k, computeU = true)
      var u = lameSVD.U
      var umat = Matrices.dense(u.numRows().toInt, u.numCols().toInt, RowMatrixToBreeze(u).data)
      var vmat = lameSVD.V
      if (vmat.numRows != k) { // transpose if necessary
        vmat = vmat.transpose
      }
      SingularValueDecomposition(umat, lameSVD.s, vmat)

    } else if (m >= n) {
      ///////////////////////////////////////////////////////
      // Step 1:
      // Generate Q matrix with values between -1 and 1.
      // Size n rows, l col
      ///////////////////////////////////////////////////////

      // Python from FBPCA
      // #
      // # Apply A to a random matrix, obtaining Q.
      // #
      // if isreal:
      //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
      // if not isreal:
      //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
      //         + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))

      // Don't worry about isreal - just do it for normal scalars

      // multiply the original data matrix with a random matrix
      // size m x n * n x bs ==> m x bs
      // multiplies uniform random matrix by 2, subtract 1 to get range -1 to 1
      var q: BDM[Double] = c * ((BDM.rand(n, bs) *:* 2.0) - 1.0)

      ////////////////////////////////////////////////////////
      // Step 2:
      // Perform the QR/LU decomposition
      ////////////////////////////////////////////////////////

      // Python from FBPCA
      // #
      // # Form a matrix Q whose columns constitute a
      // # well-conditioned basis for the columns of the earlier Q.
      // #
      // if n_iter == 0:
      //     (Q, _) = qr(Q, mode='economic')
      // if n_iter > 0:
      //     (Q, _) = lu(Q, permute_l=True)

      if (p_iter == 0) {
        // TODO: Come up with a way to calculate the QR factorization in a distributed fashion
        // Calculates and returns the Q from the QR factorization
        val qr: BQR.DenseQR = BQR.apply(q)
        q = qr.q
      }
      if (p_iter > 0) {
        // TODO: Come up with a way to calculate the LU factorization in a distributed fashion
        // See
        // https://issues.apache.org/jira/browse/SPARK-8514
        // Calculates and returns the L from the LU factorization of the q matrix
        var (pl, _) = LUFactorization(q) // computes only P * L
        q = pl
      }

      /////////////////////////////////////////////////////////
      // Step 3:
      // Run the power method for n_iter
      /////////////////////////////////////////////////////////

      // Python Code from FBPCA
      // #
      // # Conduct normalized power iterations.
      // #
      // for it in range(n_iter):

      //     Q = mult(Q.conj().T, A).conj().T

      //     (Q, _) = lu(Q, permute_l=True)

      //     Q = mult(A, Q)

      //     if it + 1 < n_iter:
      //         (Q, _) = lu(Q, permute_l=True)
      //     else:
      //         (Q, _) = qr(Q, mode='economic')

      for (i <- 0 to p_iter) {
        // We're not worried about conjugates - assume we're working with
        // real numbers
        // We need to write a transpose function
        q = (q.t * c).t

        // compute the LU factorization of Q
        var (q2: BDM[Double], _) = LUFactorization(q)
        q = c * q2

        if (i + 1 < p_iter) {
          // Compute LU
          var (q3: BDM[Double], _) = LUFactorization(q)
          q = q3
        } else {
          // Compute QR
          val qr: BQR.DenseQR = BQR.apply(q)
          q = qr.q
        }

      }


      /////////////////////////////////////////////////////////
      // Step 4:
      // SVD Q and original matrix to get singular values
      // (Assuming using BLAS?) - We should test this.
      /////////////////////////////////////////////////////////

      // # SVD Q'*A to obtain approximations to the singular values
      // # and right singular vectors of A; adjust the left singular
      // # vectors of Q'*A to approximate the left singular vectors
      // # of A.
      // #
      // QA = mult(Q.conj().T, A)
      // (R, s, Va) = svd(QA, full_matrices=False)
      // U = Q.dot(R)

      var qa = q.t * c
      // Perform local ARPACK SVD on this matrix (Or normal rowmatrix SVD?)
      val brzSvd.SVD(tempU, s, v) = brzSvd(qa)
      q = (q * tempU) //*:* -1.0


      ////////////////////////////////////////////////////////
      // Step 5:
      // Retain only the first k rows/columns and return
      ////////////////////////////////////////////////////////

      // #
      // # Retain only the leftmost k columns of U, the uppermost
      // # k rows of Va, and the first k entries of s.
      // #
      // return U[:, :k], s[:k], Va[:k, :]

      // Truncate rows of U, s, and Va
      var U = Matrices.dense(q.rows, k, Arrays.copyOfRange(q.data, 0, q.rows * k))
      var sigmas = Vectors.dense(Arrays.copyOfRange(s.data, 0, k)) // Truncated singular values
      var tv: BDM[Double] = v(0 to k - 1, ::)
      var Va = Matrices.dense(tv.rows, tv.cols, Arrays.copyOfRange(tv.data, 0, tv.rows * tv.cols))
      SingularValueDecomposition(U, sigmas, Va)

    } else if (m < n) {

      var q: BDM[Double] = (((BDM.rand(bs, m.toInt) *:* 2.0) - 1.0) * c).t

      if (p_iter == 0) {
        val qr: BQR.DenseQR = BQR.apply(q)
        q = qr.q
      } else if (p_iter > 0) {
        var (lu, _) = LUFactorization(q)
        q = lu
      }

      for (i <- 0 to p_iter) {
        q = c * q
        var (q2: BDM[Double], _) = LUFactorization(q)
        q = q2
        q = (q.t * c).t

        if (i + 1 < p_iter) {
          var (q3: BDM[Double], _) = LUFactorization(q)
          q = q3
        } else {
          val qr: BQR.DenseQR = BQR.apply(q)
          q = qr.q
        }
      }

      // Perform SVD on the smaller matrix
      var aq = c * q
      val brzSvd.SVD(u, s, tempV) = brzSvd(aq)
      q = tempV * q.t

      // Truncate SVD response
      var sigmas = Vectors.dense(Arrays.copyOfRange(s.data, 0, k)) // Truncated singular values
      var U = Matrices.dense(u.rows, k, Arrays.copyOfRange(u.data, 0, u.rows * k))
      var Va = Matrices.dense(k, q.cols, Arrays.copyOfRange(q.data, 0, q.cols * k))
      //      if (Va.numRows != k) {
      //        Va = Va.transpose
      //        System.out.println("HERE")
      //      }
      SingularValueDecomposition(U, sigmas, Va)
    } else {
      null // This should never return because we cover all other cases with m <= n and m > n
    }
  }

  def min(a: Long, b: Long): Long = if (a < b) a else b

  def RowMatrixToBreeze(A: RowMatrix): BDM[Double] = {
    val m = A.numRows().toInt
    val n = A.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    A.rows.collect().foreach { vector =>
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
      i += 1
    }
    mat
  }

  def LUFactorization(a: BDM[Double], pl_only: Boolean = true): (BDM[Double], BDM[Double]) = {

    var (d: BDM[Double], p: Array[Int]) = BLU.apply(a)
    var pmat: BDM[Double] = create_perm_mat(p.array, a.rows.toInt, a.cols.toInt)
    var pl = pmat * lowT(d)(0 to a.rows.toInt - 1, 0 to min(a.rows, a.cols).toInt - 1)

    //    println("PERMUTATION MATRIX")
    //    println(pmat)
    //    println("P * L * U")
    //    println(t2)
    if (pl_only) {
      (pl, null)
    } else {
      var ut: BDM[Double] = upT(d)(0 to min(a.rows, a.cols).toInt - 1, 0 to a.cols.toInt - 1)
      (pl, ut)
    }

  }

  // A = P * L * U
  // returns ((P * L), U)
  def create_perm_mat(ipiv: Array[Int], rows: Int, cols: Int): BDM[Double] = {
    val ipiv_dim: Int = min(rows, cols).toInt
    var parr: Array[Int] = new Array(rows)

    // 0 Index everything first

    // Create perm vector
    for (x <- 0 to rows - 1) {
      parr(x) = x + 1
    }
    for (i <- 0 to ipiv_dim - 1) {
      val i2: Int = ipiv(i) - 1
      var tmp: Int = parr(i)
      parr(i) = parr(i2)
      parr(i2) = tmp
    }

    // Create permutation matrix
    var pm: BDM[Double] = BDM.eye[Double](rows)
    for (r1 <- ipiv_dim - 1 to 0 by -1) {
      val r2: Int = ipiv(r1) - 1
      if (r1 != r2) {
        for (col <- 0 to rows - 1) {
          // Swap pm(r1, col) with pm(row2, col)
          val tmp: Double = pm(r1, col)
          pm(r1, col) = pm(r2, col)
          pm(r2, col) = tmp
        }
      }
    }
    pm
  }

  def lowT(m: BDM[Double]): BDM[Double] = {
    BDM.tabulate(m.rows, m.cols)((i, j) =>
      if (j == i) {
        1
      } else if (j < i) {
        m(i, j)
      } else {
        implicitly[Semiring[Double]].zero
      }
    )
  }

  def upT(m: BDM[Double]): BDM[Double] = {
    BDM.tabulate(m.rows, m.cols)((i, j) =>
      if (j == i) {
        m(i, j)
      } else if (j > i) {
        m(i, j)
      } else {
        implicitly[Semiring[Double]].zero
      }
    )
  }

  def max(a: Long, b: Long): Long = if (a > b) a else b

}

