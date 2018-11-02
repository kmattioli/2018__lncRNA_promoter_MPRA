note, pool1_fimo_map turned into genom coords using following command:
awk '{{OFS="\t"} if (NR>1) {match($2, /\_\_([a-zXY0-9]*)\:/, a); match($2, /([0-9]*)\.\./, b); match($2, /\.\.([0-9]*)\,/, c); match($2, /\,([\+\-])\_\_/, d)} if (d[1] == "+") {print a[1], b[1]+$3-1, b[1]+$3+length($9)-1, $2, 0, d[1], a[1], b[1]+$3-1, b[1]+$3+length($9)-1, $1, $6, $5} else {print a[1], c[1]-$4, c[1]-$4+length($9), $2, 0, d[1], a[1], c[1]-$4, c[1]-$4+length($9), $1, $6, $5}}' old/pool1_fimo_map.orig.txt > pool1_fimo_map.txt

to join the output of new_chip_intersect.txt back with the orig fimo coords:
awk '{{OFS="\t"} {print $0, $10 ":::" $4 ":::" $1 ":::" $11 ":::" $12}}' all_fimo_map.txt | sort -k13,13 > all_fimo_map.with_join_field.txt
awk '{{OFS="\t"} {print $4 ":::" $1 ":::" $5 ":::" $6}}' all_fimo_map.new_chip_intersected.txt | sort | join -1 1 -2 13 - all_fimo_map.with_join_field.txt | awk '{{OFS="\t"} {print $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13}}' > all_fimo_map.new_chip_intersected.joined.txt
(old files in old/ dir, rewrote new_chip_intersect.txt with joined.txt)

to dedupe motif files and pick only highest score of perfectly overlapping ones (i.e., +/- both map):
sort -k1,1 -k2,2n -k10,10 -k11,11rn all_fimo_map.txt | uniq | awk '{{OFS="\t"} {print $11, $12, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10}}' | uniq -f 2 | awk '{{OFS="\t"} {print $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $1, $2}}' > all_fimo_map.new_deduped.txt
