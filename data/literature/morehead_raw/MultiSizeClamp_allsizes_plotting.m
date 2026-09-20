%% Multisize clamp - all sizes plotting script
%% Ryandefaults; close all;
load data_MultiSizeClamp_1237; 
load data_MultiSizeClamp_15304560; 
load data_MultiSizeClamp_7080100110;
load data_MultiSizeClamp_125135145165;

ccw1to7 = [1 3 5 7 9 11 13 15 17];
cw1to7 = [2 4 6 8 10 12 14 16 18 19 20];
ns1to7=1:length(ccw1to7)+length(cw1to7);
ccw15to60 = [1 3 5 7 10 12 14 16 18 20];
cw15to60 = [2 4 6 8 9 11 13 15 17 19 21];
ns15to60=length(ns1to7)+1:length(ns1to7)+length(ccw15to60)+length(cw15to60);
ccw70to110 = [1 3 5 7 9 11 13 15 17];
cw70to110 = [2 4 6 8 10 12 14 16 18];
ns70to110=length(ns1to7)+length(ns15to60)+1:length(ns1to7)+length(ns15to60)+length(ccw70to110)+length(cw70to110);
ccw125to165 = [1 3];
cw125to165 = [2 4];
ns125to165=length(ns1to7)+length(ns15to60)+length(ns70to110)+1:length(ns1to7)+length(ns15to60)+length(ccw70to110)+length(cw70to110)+length(ccw125to165)+length(cw125to165);
ins=[ns1to7 ns15to60 ns70to110 ns125to165]; pp_=ones(length(ins),540,16)*NaN;

%%
ccw = [1 3 5 7 9 11 13 15 17];
cw = [2 4 6 8 10 12 14 16 18 19 20];
ns = 1:length(ccw)+length(cw);

app(cw,:) = squeeze(data_MultiSizeClamp_1237.rpospeakth(cw,:)); app(ccw,:) = squeeze(data_MultiSizeClamp_1237.rpospeakth(ccw,:))*signflip;
irot(cw,:) = squeeze(data_MultiSizeClamp_1237.rotation(cw,:))*signflip; irot(ccw,:) = squeeze(data_MultiSizeClamp_1237.rotation(ccw,:));
tangle = data_MultiSizeClamp_1237.targetangle;

thresh = 90;
app(app>thresh)=NaN; app(app<-thresh)=NaN;

clamptrials=321:1840;

rots=unique(irot(:,clamptrials)); iirot=1:length(rots);
targs=unique(tangle); itargs=1:length(targs);

rrindex=irot(:,321:324);
ttindex=tangle(:,321:324);


temp_p(1,:)=app(1,tangle(1,:)==180);
temp_p(2,:)=app(1,tangle(1,:)==90);
temp_p(3,:)=nanmean(temp_p(1:2,:),1);
app(1,tangle(1,:)==180)=temp_p(3,:);
app(1,tangle(1,:)==90)=NaN;
subfixer=[0 90 180 270];


targetpicker(1,:)=subfixer;
for s=ns(2:end)
    for i=iirot
        tindex=find(rrindex(s,:)==rots(i));
        targetpicker(s,i)=ttindex(s,tindex);
    end
end

pp=ones(length(ns),540)*NaN;


for s = ns
    for i = iirot
        rindex=find(tangle(s,:)==targetpicker(s,i));
        pp(s,1:length(rindex),i)=app(s,rindex);
    end
end

base=31:40;
for i = 1:size(pp,2); pp(:,i,:) = pp(:,i,:)-nanmean(pp(:,base,:),2); end
pp_(1:length(ns1to7),:,1:4)=pp;

clear data_MultiSizeClamp_1237 ns cw ccw pp app irot tangle rindex tindex rrindex ttindex subfixer;
%%
ccw = [1 3 5 7 10 12 14 16 18 20];
cw = [2 4 6 8 9 11 13 15 17 19 21];
ns = 1:length(ccw)+length(cw);

app(cw,:) = squeeze(data_MultiSizeClamp_15304560.rpospeakth(cw,:)); app(ccw,:) = squeeze(data_MultiSizeClamp_15304560.rpospeakth(ccw,:))*signflip;
irot(cw,:) = squeeze(data_MultiSizeClamp_15304560.rotation(cw,:))*signflip; irot(ccw,:) = squeeze(data_MultiSizeClamp_15304560.rotation(ccw,:));
tangle = data_MultiSizeClamp_15304560.targetangle;

app(app>thresh)=NaN; app(app<-thresh)=NaN;

clamptrials=321:1840;

rots=unique(irot(:,clamptrials)); iirot=1:length(rots);
targs=unique(tangle); itargs=1:length(targs);

rrindex=irot(:,321:324);
ttindex=tangle(:,321:324);

for s=ns(1:end)
    for i=iirot
        tindex=find(rrindex(s,:)==rots(i));
        targetpicker(s,i)=ttindex(s,tindex);
    end
end

pp=ones(length(ns),540)*NaN;


for s = ns
    for i = iirot
        rindex=find(tangle(s,:)==targetpicker(s,i));
        pp(s,1:length(rindex),i)=app(s,rindex);
    end
end

base=31:40;
for i = 1:size(pp,2); pp(:,i,:) = pp(:,i,:)-nanmean(pp(:,base,:),2); end
pp_(1:length(ns15to60),:,5:8)=pp; 

clear data_MultiSizeClamp_15304560 ns cw ccw pp app irot tangle rindex tindex rrindex ttindex subfixer;
%%
ccw = [1 3 5 7 9 11 13 15 17];
cw = [2 4 6 8 10 12 14 16 18];
ns = 1:length(ccw)+length(cw);

app(cw,:) = squeeze(data_MultiSizeClamp_7080100110.rpospeakth(cw,:)); app(ccw,:) = squeeze(data_MultiSizeClamp_7080100110.rpospeakth(ccw,:))*signflip;
irot(cw,:) = squeeze(data_MultiSizeClamp_7080100110.rotation(cw,:))*signflip; irot(ccw,:) = squeeze(data_MultiSizeClamp_7080100110.rotation(ccw,:));
tangle = data_MultiSizeClamp_7080100110.targetangle;

app(app>thresh)=NaN; app(app<-thresh)=NaN;
clamptrials=321:1840;

rots=unique(irot(:,clamptrials)); iirot=1:length(rots);
targs=unique(tangle); itargs=1:length(targs);

rrindex=irot(:,321:324);
ttindex=tangle(:,321:324);

for s=ns(1:end)
    for i=iirot
        tindex=find(rrindex(s,:)==rots(i));
        targetpicker(s,i)=ttindex(s,tindex);
    end
end

pp=ones(length(ns),540)*NaN;


for s = ns
    for i = iirot
        rindex=find(tangle(s,:)==targetpicker(s,i));
        pp(s,1:length(rindex),i)=app(s,rindex);
    end
end

base=31:40;
for i = 1:size(pp,2); pp(:,i,:) = pp(:,i,:)-nanmean(pp(:,base,:),2); end
pp_(1:length(ns70to110),:,9:12)=pp; 

clear data_MultiSizeClamp_7080100110 ns cw ccw pp app irot tangle rindex tindex rrindex ttindex subfixer;
%%
ccw = [1 3];
cw = [2 4];
ns = 1:length(ccw)+length(cw);

app(cw,:) = squeeze(data_MultiSizeClamp_125135145165.rpospeakth(cw,:)); app(ccw,:) = squeeze(data_MultiSizeClamp_125135145165.rpospeakth(ccw,:))*signflip;
irot(cw,:) = squeeze(data_MultiSizeClamp_125135145165.rotation(cw,:))*signflip; irot(ccw,:) = squeeze(data_MultiSizeClamp_125135145165.rotation(ccw,:));
tangle = data_MultiSizeClamp_125135145165.targetangle;

thresh = 90;
app(app>thresh)=NaN; app(app<-thresh)=NaN;
clamptrials=321:1840;

rots=unique(irot(:,clamptrials)); iirot=1:length(rots);
targs=unique(tangle); itargs=1:length(targs);

rrindex=irot(:,321:324);
ttindex=tangle(:,321:324);

for s=ns(1:end)
    for i=iirot
        tindex=find(rrindex(s,:)==rots(i));
        targetpicker(s,i)=ttindex(s,tindex);
    end
end

pp=ones(length(ns),540)*NaN;


for s = ns
    for i = iirot
        rindex=find(tangle(s,:)==targetpicker(s,i));
        pp(s,1:length(rindex),i)=app(s,rindex);
    end
end

base=31:40;
for i = 1:size(pp,2); pp(:,i,:) = pp(:,i,:)-nanmean(pp(:,base,:),2); end
pp_(1:length(ns125to165),:,13:16)=pp;

clear data_MultiSizeClamp_125135145165 ns cw ccw pp app irot tangle rindex tindex rrindex ttindex subfixer;
%%
tnt=1:540;


initialtrials=[-1:1]+45;
initial=squeeze(nanmean(pp_(:,initialtrials,:),2))./5;
minitial=nanmean(initial,1);
% steinitial=nanstd(initial,[],1);
steinitial1to7=nanste(initial(:,1:4));steinitial15to60=nanste(initial(1:length(ns15to60),5:8));steinitial70to110=nanste(initial(1:length(ns70to110),9:12)); steinitial125to165=nanste(initial(1:length(ns125to165),13:16));
steinitial(1:4)=steinitial1to7; steinitial(5:8)=steinitial15to60; steinitial(9:12)=steinitial70to110; steinitial(13:16)=steinitial125to165;

totals=squeeze(nanmean(pp_(:,301:500,:),2));
mtotals=nanmean(totals,1);
% stetotals=nanstd(totals,[],1);
stetotals1to7=nanste(totals(:,1:4));stetotals15to60=nanste(totals(1:length(ns15to60),5:8));stetotals70to110=nanste(totals(1:length(ns70to110),9:12)); stetotals125to165=nanste(totals(1:length(ns125to165),13:16));
stetotals(1:4)=stetotals1to7; stetotals(5:8)=stetotals15to60; stetotals(9:12)=stetotals70to110; stetotals(13:16)=stetotals125to165;


actualpos_=[1.75 2.5 3.5 7.5]; actualpos=actualpos_;
actualpos_=[15 30 45 60]; actualpos=[actualpos actualpos_];
actualpos_=[70 80 100 110]; actualpos=[actualpos actualpos_];
actualpos_=[125 135 145 165]; actualpos=[actualpos actualpos_];


individualsubs=0;

figure(1); clf; hold on;
if individualsubs==1
plot(actualpos(1:4),initial(:,1:4),'bo-');
plot(actualpos(5:8),initial(:,5:8),'bo-');
plot(actualpos(9:12),initial(:,9:12),'bo-');
plot(actualpos(13:16),initial(:,13:16),'bo-');
end
shadedErrorBar(actualpos(:,1:4),minitial(:,1:4),steinitial(:,1:4),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,5:8),minitial(:,5:8),steinitial(:,5:8),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,9:12),minitial(:,9:12),steinitial(:,9:12),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,13:16),minitial(:,13:16),steinitial(:,13:16),{'-','color','b','linewidth',lw},2);
axis([-5 180 -0.5 3]);
set(gca,'xtick',[1.75 2.5 3.5 7.5 15 30 45 60 70 80 100 110 125 135 145 165 180],'xticklabel',{'1','2','3°','7.5°','15°','30°','45°','60°','70°','80°','100°','110°','125°','135°','145°','165°','180°'});
xlabel('Clamp Offset Angle'); ylabel('Hand Angle @ 5 cycles');
line([-5 180],[0 0],'linestyle','-','color','k');
ylabel('Hand Angle @ 5 cycles');
title('Initial Learning');
grid on;

figure(2); clf; hold on;
if individualsubs ==1
plot(actualpos(1:4),totals(:,1:4),'bo-');
plot(actualpos(5:8),totals(:,5:8),'bo-');
plot(actualpos(9:12),totals(:,9:12),'bo-');
plot(actualpos(13:16),totals(:,13:16),'bo-');
end
shadedErrorBar(actualpos(:,1:4),mtotals(:,1:4),stetotals(:,1:4),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,5:8),mtotals(:,5:8),stetotals(:,5:8),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,9:12),mtotals(:,9:12),stetotals(:,9:12),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,13:16),mtotals(:,13:16),stetotals(:,13:16),{'-','color','b','linewidth',lw},2);
axis([-5 180 -10 60]);
set(gca,'xtick',[1.75 2.5 3.5 7.5 15 30 45 60 70 80 100 110 125 135 145 165 180],'xticklabel',{'1','2','3°','7.5°','15°','30°','45°','60°','70°','80°','100°','110°','125°','135°','145°','165°','180°'});
xlabel('Clamp Offset Angle'); ylabel('Hand Angle (deg)');
line([-5 180],[0 0],'linestyle','-','color','k');
ylabel('Late Hand Angle (last 200 cycles)');
title('Asymptotic Learning');
grid on;


% axis([-5 175 -10 60]);
% set(gca,'xtick',[1.75 2.5 3.5 7.5 15 30 45 60 70 80 100 110 125 135 145 165],'xticklabel',{'1','2','3°','7.5°','15°','30°','45°','60°','70°','80°','100°','110°','125°','135°','145°','165°'});

figure(3); clf; hold on;
% if individualsubs ==1
% plot(actualpos(1:4),totals(:,1:4)./actualpos(1:4),'bo-');
% plot(actualpos(5:8),totals(:,5:8)./actualpos(5:8),'bo-');
% plot(actualpos(9:12),totals(:,9:12)./actualpos(9:12),'bo-');
% plot(actualpos(13:16),totals(:,13:16)./actualpos(13:16),'bo-');
% end
shadedErrorBar(actualpos(:,1:4),mtotals(:,1:4)./actualpos(1:4),stetotals(:,1:4)./actualpos(1:4),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,5:8),mtotals(:,5:8)./actualpos(5:8),stetotals(:,5:8)./actualpos(5:8),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,9:12),mtotals(:,9:12)./actualpos(9:12),stetotals(:,9:12)./actualpos(9:12),{'-','color','b','linewidth',lw},2);
shadedErrorBar(actualpos(:,13:16),mtotals(:,13:16)./actualpos(13:16),stetotals(:,13:16)./actualpos(13:16),{'-','color','b','linewidth',lw},2);
plot(actualpos,actualpos.^-.28,'k-')
axis([-5 180 -2 12]);
set(gca,'xtick',[1.75 2.5 3.5 7.5 15 30 45 60 70 80 100 110 125 135 145 165 180],'xticklabel',{'1','2','3°','7.5°','15°','30°','45°','60°','70°','80°','100°','110°','125°','135°','145°','165°','180°'});
xlabel('Clamp Offset Angle'); ylabel('Hand Angle (deg)');
line([-5 180],[0 0],'linestyle','-','color','k');
ylabel('Late Hand Angle (last 200 cycles)');
title('Error Sensitivity');
grid on;


















