function e = relative_error(a,b)
%RELATIVE_ERROR ||a-b||_2 / ||a||_2.
e = norm(a-b)/(norm(a)+eps);
end
