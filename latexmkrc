$clean_ext .= ' %R.figlist %R-figure* %R.makefile fls.tmp';
$latex    = 'internal tikzlatex latex    %B %O %S';
$pdflatex = 'internal tikzlatex pdflatex %B %O %S';
$lualatex = 'internal tikzlatex lualatex %B %O %S';
$xelatex  = 'internal tikzlatex xelatex  %B %O %S';
$hash_calc_ignore_pattern{'pdf'} = '^(/CreationDate|/ModDate|/ID)';
$hash_calc_ignore_pattern{'ps'} = '^%%CreationDate';

sub pre_compile {
    if ($^O eq 'MSWin32') {  # For Windows systems
        system "mkdir .\\tmp";
        system "mkdir .\\tmp\\tmp";
        system "mkdir .\\tmp\\tex";
    } else {  # For Unix-like systems
        system "mkdir -p ./tmp";
        system "mkdir -p ./tmp/tmp";
        system "mkdir -p ./tmp/tex";
    }
};

pre_compile();

sub tikzlatex {
    my ($engine, $base, @args) = @_;
    my $ret = 0;
    if ($^O eq 'MSWin32') { 
        # For Windows systems
        print "Tikzlatex: ===Running '$engine @args'...\n";
        $ret = system( $engine, @args );
        print "Tikzlatex: Fixing .fls file ...\n";
        system "echo INPUT \"$aux_dir1$base.figlist\"  >  \"$aux_dir1$base.fls.tmp\"";
        system "echo INPUT \"$aux_dir1$base.makefile\" >> \"$aux_dir1$base.fls.tmp\"";

        system "type \"$aux_dir1$base.fls\" >> \"$aux_dir1$base.fls.tmp\"";

        rename "$aux_dir1$base.fls.tmp", "$aux_dir1$base.fls";

        if ($ret) { return $ret; }
            if ( -e "$aux_dir1$base.makefile" ) {
                if ($engine eq 'xelatex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under xelatex\n";
                    system("sed \"s/^\^\^I/\t/\" $aux_dir1$base.makefile > $aux_dir1$base.makefile.tmp");
                    rename "$aux_dir1$base.makefile.tmp", "$aux_dir1$base.makefile";
                }
                elsif ($engine eq 'latex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under latex\n";
                    system("sed \"s/\\.epsi/\\.ps/\" $aux_dir1$base.makefile > $aux_dir1$base.makefile.tmp");
                    rename "$aux_dir1$base.makefile.tmp", "$aux_dir1$base.makefile";
                }
                print "Tikzlatex: ---Running 'make -f $aux_dir1$base.makefile' ...\n";
                if ($aux_dir) {
                    system("sed \"s#$base.figlist#$aux_dir1$base.figlist#g\" $aux_dir1$base.makefile > $aux_dir1$base.makefile.tmp");
                    rename "$aux_dir1$base.makefile.tmp", "$aux_dir1$base.makefile";
                    system("sed \"s/mkdir -p/mkdir/\" $aux_dir1$base.makefile > $aux_dir1$base.makefile.tmp");
                    rename "$aux_dir1$base.makefile.tmp", "$aux_dir1$base.makefile";

                    system "cp $aux_dir1$aux_dir1*.md5 $aux_dir1";

                    system "rm -rf $aux_dir1$aux_dir1";
                    $ret = system "sh make",  "-j", "10", "-f", "$aux_dir1$base.makefile";
                    system "rm $base.run.xml";
                }
                else {
                    $ret = system "make",  "-j", "10", "-f", "$base.makefile";
                }
                if ($ret) {
                    print "Tikzlatex: !!!!!!!!!!!!!! Error from make !!!!!!!!! \n",
                        "  The log files for making the figures '$aux_dir1$base-figure*.log'\n",
                        "  may have information\n";
                }
            }
        else {
            print "Tikzlatex: No '$aux_dir1$base.makefile', so I won't run make.\n";
        }
        return $ret;
    } elsif ($^O eq 'linux') {  # For linux systems
        print "Tikzlatex: ===Running '$engine @args'...\n";
        $ret = system( $engine, @args );
        print "Tikzlatex: Fixing .fls file ...\n";
        system "echo INPUT \"$aux_dir1$base.figlist\"  >  \"$aux_dir1$base.fls.tmp\"";
        system "echo INPUT \"$aux_dir1$base.makefile\" >> \"$aux_dir1$base.fls.tmp\"";

        system "cat \"$aux_dir1$base.fls\" >> \"$aux_dir1$base.fls.tmp\"";

        rename "$aux_dir1$base.fls.tmp", "$aux_dir1$base.fls";

        if ($ret) { return $ret; }
            if ( -e "$aux_dir1$base.makefile" ) {
                if ($engine eq 'xelatex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under xelatex\n";
                    system('sed', '-i', 's/^\^\^I/\t/g', "$aux_dir1$base.makefile");
                }
                elsif ($engine eq 'latex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under latex\n";
                    system('sed', '-i', 's/\.epsi/\.ps/g', "$aux_dir1$base.makefile");
                }
                print "Tikzlatex: ---Running 'make -f $aux_dir1$base.makefile' ...\n";
                if ($aux_dir) {
                    system("sed -i 's|$base.figlist|$aux_dir1$base.figlist|g' $aux_dir1$base.makefile");
                    system "cp $aux_dir1$aux_dir1*.md5 $aux_dir1";
                    system "rm -rf $aux_dir1$aux_dir1";
                    $ret = system "make",  "-j", "128", "-f", "$aux_dir1$base.makefile";
                    system "rm $base.run.xml";
                }
                else {
                    $ret = system "make",  "-j", "128", "-f", "$base.makefile";
                }
                if ($ret) {
                    print "Tikzlatex: !!!!!!!!!!!!!! Error from make !!!!!!!!! \n",
                        "  The log files for making the figures '$aux_dir1$base-figure*.log'\n",
                        "  may have information\n";
                }
            }
        else {
            print "Tikzlatex: No '$aux_dir1$base.makefile', so I won't run make.\n";
        }
        return $ret;
    }
    else {  # For Unix-like systems
        print "Tikzlatex: ===Running '$engine @args'...\n";
        $ret = system( $engine, @args );
        print "Tikzlatex: Fixing .fls file ...\n";
        system "echo INPUT \"$aux_dir1$base.figlist\"  >  \"$aux_dir1$base.fls.tmp\"";
        system "echo INPUT \"$aux_dir1$base.makefile\" >> \"$aux_dir1$base.fls.tmp\"";

        system "cat \"$aux_dir1$base.fls\" >> \"$aux_dir1$base.fls.tmp\"";

        rename "$aux_dir1$base.fls.tmp", "$aux_dir1$base.fls";

        if ($ret) { return $ret; }
            if ( -e "$aux_dir1$base.makefile" ) {
                if ($engine eq 'xelatex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under xelatex\n";
                    system('sed', '-i', '', 's/^\^\^I/\t/', "$aux_dir1$base.makefile");
                }
                elsif ($engine eq 'latex') {
                    print "Tikzlatex: ---Correcting '$aux_dir1$base.makefile' made under latex\n";
                    system('sed', '-i', '', 's/\.epsi/\.ps/', "$aux_dir1$base.makefile");
                }
                print "Tikzlatex: ---Running 'make -f $aux_dir1$base.makefile' ...\n";
                if ($aux_dir) {
                    system('sed', '-i', '', "s#$base.figlist#$aux_dir1$base.figlist#g", "$aux_dir1$base.makefile");
                    system "cp $aux_dir1$aux_dir1*.md5 $aux_dir1";
                    system "rm -rf $aux_dir1$aux_dir1";
                    $ret = system "make",  "-j", "10", "-f", "$aux_dir1$base.makefile";
                    system "rm $base.run.xml";
                }
                else {
                    $ret = system "make",  "-j", "10", "-f", "$base.makefile";
                }
                if ($ret) {
                    print "Tikzlatex: !!!!!!!!!!!!!! Error from make !!!!!!!!! \n",
                        "  The log files for making the figures '$aux_dir1$base-figure*.log'\n",
                        "  may have information\n";
                }
            }
        else {
            print "Tikzlatex: No '$aux_dir1$base.makefile', so I won't run make.\n";
        }
        return $ret;
    }
}