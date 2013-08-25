run: games

%.dot.pdf: %.dot
	dot -Tpdf -O $< || rm $@

%: %.rs
	rustc --cfg debug -Z debug-info -Aunused-mut $<
