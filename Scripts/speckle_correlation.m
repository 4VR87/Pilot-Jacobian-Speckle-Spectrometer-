function c = speckle_correlation(a,b)
%SPECKLE_CORRELATION Pearson-like zero-mean cosine similarity.
a = a(:)-mean(a(:)); b = b(:)-mean(b(:));
c = (a'*b)/(norm(a)*norm(b)+eps);
end
