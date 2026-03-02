close all
clear all
format longG
set(0, 'DefaultTextInterpreter', 'latex'); % Per il testo
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex'); % Per la legenda

%IMPORT DATI
sign =["p" "m"];
% il ciclo for doppio serve ad importare allo stesso tempo i dati del
% guadagno sia su Vc negativo che positivo
for j=1:2
    for i =1:5
        data = readmatrix(sprintf('Resonance-sine%s2(%d).txt',sign(j),i));
        if j==1
            gainp(:,i) = data(:,2);
            phip(:,i)= data(:,3);
            fp= data(:,1);
        else
            gainm(:,i) = data(:,2);
            phim(:,i)= data(:,3);
            fm = data(:,1);

        end
    end
end

%medie e dev std
gainmm = mean(gainm,2);  
sigmagm = std(gainm')';
gainpm = mean(gainp,2);  
sigmagp = std(gainp')';

% FIT SUI GUADAGNI
G =@(p,x)  (p(3)+p(4)./(sqrt((1-(x./p(1)).^2).^2+(p(2).*x.*2.*pi).^2)));
w0 = 7e3;
tau = 1.78e-5;
offs = 3e-2;
cm = 0.98;
pp0 = [w0 tau offs cm];

%punto di eq negativo
[ppm,R,~,covbm,chi2nm] = nlinfit(fm,gainmm,G,pp0,Weights=sigmagm.^(-2));
dppm = sqrt(diag(covbm));

%punto di eq positivo
[ppp,R,~,covbp,chi2np] = nlinfit(fp,gainpm,G,pp0,Weights=sigmagp.^(-2));
dppp = sqrt(diag(covbp));

figure(1)
sgtitle('Plot di Bode sul punto di equilibrio a -$2V^*$')
subplot(2,1,1)
errorbar(fm,gainmm,sigmagm, '.', CapSize=0.2)
hold on 
plot(fm,G(ppm,fm));
hold off
ylabel('Guadagno')
xscale log
yscale log
legend('Punti sperimentali', 'Fit', Location='SW')
grid on
subtitle('Forzante = 0.1 V')

%residui -
res = (gainmm - G(ppm, fm)); %calcolo dei residui
resn = res./sigmagm;
i = ones(length(gainmm));
subplot(2,1,2) %plot dei residui
errorbar(fm, resn, i, '.',MarkerSize=8,Color="blue")
xscale log
yline(0, Color='black', LineStyle='--')
xlabel('Frequenza [Hz]')
ylabel("Residui normalizzati")
grid on

%acquisizione immagine
myfig = gcf;
exportgraphics(gcf,"bodeeqnegativosecondo01.pdf",ContentType="vector");

figure(2)
sgtitle('Plot di Bode sul punto di equilibrio a +$2V^*$')
subplot(2,1,1)

errorbar(fp,gainpm,sigmagp,  '.', CapSize=0.2)
hold on 
plot(fp,G(ppp,fp));
hold off
ylabel('Guadagno')
xscale log
yscale log
legend('Punti sperimentali', 'Fit', Location='SW')
grid on
subtitle('Forzante = 0.1 V')

%residui +
res = (gainpm - G(ppp, fp)); %calcolo dei residui
resn = res./sigmagp;
i = ones(length(gainpm));
subplot(2,1,2) %plot dei residui
errorbar(fp, resn, i, '.',MarkerSize=8,Color="blue")
xscale log
yline(0, Color='black', LineStyle='--')
xlabel('Frequenza [Hz]')
ylabel("Residui normalizzati")
grid on

%acquisizione immagine
myfig = gcf;
exportgraphics(gcf,"bodeeqpositivosecondo01.pdf",ContentType="vector");

% RISULTATI DEL FIT
paramNames = {'f0', 'gamma', 'offs', 'cost'};
fprintf('\n\nFIT SU PT. EQ. NEGATIVO:\n');
for i=1:4
    fprintf('%s = %s +- %s\n', paramNames{i}, ppm(i), dppm(i))
end
fprintf('chi2n = %s', chi2nm)

fprintf('\n\nFIT SU PT. EQ. POSITIVO:\n');
for i=1:4
    fprintf('%s = %s +- %s\n', paramNames{i}, ppp(i), dppp(i))
end
fprintf('chi2n = %s\n', chi2np)





