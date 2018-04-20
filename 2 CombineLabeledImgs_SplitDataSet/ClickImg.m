function ClickImg(src, evnt)
             switch evnt.Character
                 case char(48) % background, no spores 
                     set(src, 'UserData', 0);
                 case char(49) % single spores
                     set(src, 'UserData', 1);
                 case char(50) % double spores
                     set(src, 'UserData', 2);
                 case char(51) % multiple spores
                     set(src, 'UserData', 3);
                 case char(52) % cornered truncated spores
                     set(src, 'UserData', 4);
                 case  char(98)% Bad images
                     set(src, 'UserData', -1);
                 case char(116)% others (e.g. cornered/truncated and complete in the same img)
                     set(src, 'UserData', -2);
             end
                     
end
    